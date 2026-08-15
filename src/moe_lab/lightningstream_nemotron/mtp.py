"""S10-A: the multi-token-prediction block of Nemotron 3.5 Lightning.

One `mtp.layers.0` attention mixer plus one `mtp.layers.1` MoE mixer, wired
exactly as `s10a0_mtp_structure.json` inventoried them.  There is no MTP
embedding table and no MTP LM head: both are borrowed from the backbone, so this
class holds a `LightningRuntime` and reads `rt.embed_host` / `rt.lm_head_*`
directly.

This exists to MEASURE the acceptance rate, not to decode with.  The 128 routed
experts are BF16 and are made device-resident here because that keeps the
measurement simple; in a built system there is no VRAM for that (N5/N7-B measured
0.000 GiB free) and they would have to be streamed or traded against cache slots.
Nothing timed in this file may be restated as a throughput figure.

Two wiring details are not derivable from any artefact -- transformers 5.15.0
ignores `mtp.*` entirely -- and are therefore parameters here, resolved
empirically in phase A1:

  concat_order : "eh"  -> eh_proj( [enorm(embed) ; hnorm(h)] )
                 "he"  -> eh_proj( [hnorm(h) ; enorm(embed)] )
  the caller chooses which backbone hidden it passes as `h_prev`.
"""

from __future__ import annotations

import numpy as np

MTP_LAYER0 = "mtp.layers.0"
MTP_LAYER1 = "mtp.layers.1"


class MTPBlock:
    def __init__(self, rt, max_ctx: int = 8192, concat_order: str = "eh",
                 resident_experts: bool = True, verbose: bool = True):
        if concat_order not in ("eh", "he"):
            raise ValueError(f"concat_order must be 'eh' or 'he', got {concat_order!r}")
        cp = rt.cp
        self.cp = cp
        self.rt = rt
        self.k = rt.k
        self.fused = rt.fused
        self.index = rt.index
        self.verbose = verbose
        self.concat_order = concat_order
        self.max_ctx = max_ctx

        c = rt.cfg
        self.hidden = c["hidden_size"]
        self.eps = c["layer_norm_epsilon"]
        self.top_k = c["num_experts_per_tok"]
        self.n_experts = c["n_routed_experts"]
        self.scaling = c["routed_scaling_factor"]
        self.moe_inter = c["moe_intermediate_size"]
        self.shared_inter = c["moe_shared_expert_intermediate_size"]
        self.n_heads = c["num_attention_heads"]
        self.n_kv = c["num_key_value_heads"]
        self.head_dim = c["head_dim"]
        self.groups = self.n_heads // self.n_kv
        self.kv_dim = self.n_kv * self.head_dim
        self.vocab = c["vocab_size"]

        self._relu2 = cp.ElementwiseKernel(
            "float32 x", "float32 y",
            "const float v = x > 0.0f ? x : 0.0f; y = v * v;",
            "mtp_relu2")

        self._load_skeleton()
        self._load_experts(resident_experts)
        self._alloc()
        self._init_active_vocab()

    # ------------------------------------------------------------------ load
    def _u16(self, name):
        return self.cp.asarray(self.index.read_raw(name).view(np.uint16))

    def _f32(self, name):
        return self.cp.asarray(self.index.get_float32(name).astype(np.float32).ravel())

    def _load_skeleton(self):
        self.enorm = self._u16(f"{MTP_LAYER0}.enorm.weight")
        self.hnorm = self._u16(f"{MTP_LAYER0}.hnorm.weight")
        self.eh_proj = self._u16(f"{MTP_LAYER0}.eh_proj.weight")
        self.n0_norm = self._u16(f"{MTP_LAYER0}.norm.weight")
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, n, self._u16(f"{MTP_LAYER0}.mixer.{n}.weight"))
        self.n1_norm = self._u16(f"{MTP_LAYER1}.norm.weight")
        self.gate_w = self._f32(f"{MTP_LAYER1}.mixer.gate.weight")
        self.gate_b = self._f32(f"{MTP_LAYER1}.mixer.gate.e_score_correction_bias")
        self.sh_up = self._u16(f"{MTP_LAYER1}.mixer.shared_experts.up_proj.weight")
        self.sh_dn = self._u16(f"{MTP_LAYER1}.mixer.shared_experts.down_proj.weight")
        self.final_ln = self._u16(f"{MTP_LAYER1}.final_layernorm.weight")

        for prefix in (f"{MTP_LAYER0}.eh_proj", f"{MTP_LAYER1}.mixer.experts.0.up_proj"):
            kind = self.index.quant_kind(prefix)
            if kind != "bf16":
                raise ValueError(f"{prefix}: expected bf16, checkpoint says {kind}")

    def _load_experts(self, resident: bool):
        cp = self.cp
        n, inter, hid = self.n_experts, self.moe_inter, self.hidden
        self.resident = resident
        if not resident:
            raise NotImplementedError("only device-resident experts are implemented")
        self.exp_up = cp.zeros(n * inter * hid, dtype=cp.uint16)
        self.exp_dn = cp.zeros(n * hid * inter, dtype=cp.uint16)
        self.up_stride = inter * hid
        self.dn_stride = hid * inter
        for e in range(n):
            pre = f"{MTP_LAYER1}.mixer.experts.{e}"
            up = self.index.read_raw(f"{pre}.up_proj.weight").view(np.uint16)
            dn = self.index.read_raw(f"{pre}.down_proj.weight").view(np.uint16)
            if up.size != self.up_stride or dn.size != self.dn_stride:
                raise ValueError(f"expert {e}: unexpected size {up.size}/{dn.size}")
            self.exp_up[e * self.up_stride:(e + 1) * self.up_stride] = cp.asarray(up)
            self.exp_dn[e * self.dn_stride:(e + 1) * self.dn_stride] = cp.asarray(dn)
            if self.verbose and (e + 1) % 32 == 0:
                print(f"    mtp experts resident: {e + 1}/{n}", flush=True)

    def _alloc(self):
        cp = self.cp
        self.cat = cp.zeros(2 * self.hidden, dtype=cp.float32)
        self.x = cp.zeros(self.hidden, dtype=cp.float32)
        self.normed = cp.zeros(self.hidden, dtype=cp.float32)
        self.emb = cp.zeros(self.hidden, dtype=cp.float32)
        self.acc = cp.zeros(self.hidden, dtype=cp.float32)
        self.tmp = cp.zeros(self.hidden, dtype=cp.float32)
        self.moe_out = cp.zeros(self.hidden, dtype=cp.float32)
        self.act = cp.zeros(max(self.moe_inter, self.shared_inter), dtype=cp.float32)
        self.qv = cp.zeros(self.n_heads * self.head_dim, dtype=cp.float32)
        self.kv_ = cp.zeros(self.kv_dim, dtype=cp.float32)
        self.vv = cp.zeros(self.kv_dim, dtype=cp.float32)
        self.ctx = cp.zeros(self.n_heads * self.head_dim, dtype=cp.float32)
        # FP32 KV for the MTP: it is small, and it keeps the acceptance figure
        # free of a second quantisation the built system might not use.
        self.kc = cp.zeros(self.max_ctx * self.kv_dim, dtype=cp.float32)
        self.vc = cp.zeros(self.max_ctx * self.kv_dim, dtype=cp.float32)
        self.rlog = cp.zeros(self.n_experts, dtype=cp.float32)
        self.logits = cp.zeros(self.vocab, dtype=cp.float32)
        splits = self.k.MAX_SPLITS * 4
        self.part_acc = cp.zeros(self.n_heads * splits * self.head_dim, dtype=cp.float32)
        self.part_ml = cp.zeros(self.n_heads * splits * 2, dtype=cp.float32)

    def reset(self):
        self.kc.fill(0)
        self.vc.fill(0)

    # ------------------------------------------------- active vocabulary (K2)
    def _init_active_vocab(self):
        """Scratch for a MicroSpec-style context-local draft vocabulary.

        The backbone lm_head is NVFP4: codes are [vocab, hidden/2] bytes and
        scales [vocab, hidden/16] bytes, both row-major, so a row subset is a
        gather on axis 0 and the same fused GEMV runs on the compacted head.
        Shapes are read from the checkpoint rather than assumed.
        """
        cp = self.cp
        self.active_n = None
        self.active_idx = None
        if self.rt.lm_head_kind != "nvfp4":
            self.code_row = self.scale_row = None
            return
        ce = self.index.entries["lm_head.weight"]
        se = self.index.entries["lm_head.weight_scale"]
        if ce.shape[0] != self.vocab or se.shape[0] != self.vocab:
            raise ValueError(f"lm_head rows {ce.shape} / {se.shape} != vocab {self.vocab}")
        self.code_row = ce.shape[1]
        self.scale_row = se.shape[1]
        self._codes2d = self.rt.lm_head_codes.reshape(self.vocab, self.code_row)
        self._scales2d = self.rt.lm_head_scales.reshape(self.vocab, self.scale_row)
        self.sub_logits = cp.zeros(self.vocab, dtype=cp.float32)

    def set_active_vocab(self, backbone_logits, n: int | None):
        """Pick the top-n rows of the backbone's own logits as the draft vocab.

        Called once per committed position, not per draft, so its cost is paid
        once per round. n=None restores the full head.
        """
        cp = self.cp
        if n is None:
            self.active_n = None
            self.active_idx = None
            return
        if self.code_row is None:
            raise RuntimeError("active vocabulary needs an NVFP4 lm_head")
        idx = cp.argpartition(-backbone_logits, n)[:n]
        idx = cp.sort(idx)                     # deterministic row order
        self.active_idx = idx
        self.active_n = int(n)
        self.sub_codes = cp.ascontiguousarray(self._codes2d[idx]).reshape(-1)
        self.sub_scales = cp.ascontiguousarray(self._scales2d[idx]).reshape(-1)

    def _head(self):
        """Project the final-normed state to draft logits and return argmax."""
        cp, rt = self.cp, self.rt
        if self.active_n is not None:
            self.fused.gemv_into(self.sub_logits[:self.active_n], self.sub_codes,
                                 self.sub_scales, self.normed, rt.lm_head_g,
                                 self.active_n, self.hidden)
            local = int(cp.argmax(self.sub_logits[:self.active_n]))
            return int(self.active_idx[local])
        if rt.lm_head_kind == "nvfp4":
            self.fused.gemv_into(self.logits, rt.lm_head_codes, rt.lm_head_scales,
                                 self.normed, rt.lm_head_g, self.vocab, self.hidden)
        else:
            self.k.mv_bf16(self.logits, rt.lm_head, self.normed, self.vocab, self.hidden)
        return int(cp.argmax(self.logits))

    # --------------------------------------------------------------- pieces
    def _embed(self, token_id: int):
        cp = self.cp
        if self.rt.embed_on_host:
            row = cp.asarray(
                self.rt.embed_host[token_id * self.hidden:(token_id + 1) * self.hidden])
        else:
            row = self.rt.embed[token_id * self.hidden:(token_id + 1) * self.hidden]
        self.emb[:] = (row.astype(cp.uint32) << cp.uint32(16)).view(cp.float32)

    def _route(self):
        cp = self.cp
        self.k.mv_f32(self.rlog, self.gate_w, self.normed, self.n_experts, self.hidden)
        scores = 1.0 / (1.0 + cp.exp(-self.rlog))
        choice = scores + self.gate_b
        idx = cp.argsort(-choice)[: self.top_k]
        w = scores[idx]
        w = w / (w.sum() + 1e-20) * self.scaling
        return cp.asnumpy(idx).astype(int), cp.asnumpy(w).astype(np.float64)

    def _moe(self):
        k = self.k
        idx, w = self._route()
        self.moe_out.fill(0)
        k.mv_bf16(self.act[:self.shared_inter], self.sh_up, self.normed,
                  self.shared_inter, self.hidden)
        self._relu2(self.act[:self.shared_inter], self.act[:self.shared_inter])
        k.mv_bf16(self.moe_out, self.sh_dn, self.act[:self.shared_inter],
                  self.hidden, self.shared_inter)
        for s, e in enumerate(idx):
            e = int(e)
            k.mv_bf16(self.act[:self.moe_inter],
                      self.exp_up[e * self.up_stride:(e + 1) * self.up_stride],
                      self.normed, self.moe_inter, self.hidden)
            self._relu2(self.act[:self.moe_inter], self.act[:self.moe_inter])
            k.mv_bf16(self.tmp,
                      self.exp_dn[e * self.dn_stride:(e + 1) * self.dn_stride],
                      self.act[:self.moe_inter], self.hidden, self.moe_inter)
            self.fused.accumulate_into(self.moe_out, self.tmp, float(w[s]), self.hidden)
        return idx, w

    # -------------------------------------------------------------- forward
    def forward(self, token_id: int, h_prev, pos: int):
        """One MTP step at sequence position `pos`.

        `h_prev` may alias `self.x` or `self.normed`: both are read into `cat`
        before either is overwritten, which is what makes chained drafting work
        without an extra copy.  Returns (draft_token, x, y) where `x` is the MTP
        residual stream and `y` is `final_layernorm(x)`, the two candidates for
        the next link in a chain.
        """
        cp, k = self.cp, self.k
        self._embed(token_id)
        lo, hi = self.cat[:self.hidden], self.cat[self.hidden:]
        if self.concat_order == "eh":
            e_dst, h_dst = lo, hi
        else:
            e_dst, h_dst = hi, lo
        k.norm(e_dst, self.emb, self.enorm, self.hidden, self.eps)
        k.norm(h_dst, h_prev, self.hnorm, self.hidden, self.eps)
        k.mv_bf16(self.x, self.eh_proj, self.cat, self.hidden, 2 * self.hidden)

        # ---- layers.0: attention (no RoPE, matching the backbone convention)
        k.norm(self.normed, self.x, self.n0_norm, self.hidden, self.eps)
        k.mv_bf16(self.qv, self.q_proj, self.normed, self.n_heads * self.head_dim,
                  self.hidden)
        k.mv_bf16(self.kv_, self.k_proj, self.normed, self.kv_dim, self.hidden)
        k.mv_bf16(self.vv, self.v_proj, self.normed, self.kv_dim, self.hidden)
        k.kv_write(self.kc, self.kv_, pos, self.n_kv, self.head_dim, self.max_ctx)
        k.kv_write(self.vc, self.vv, pos, self.n_kv, self.head_dim, self.max_ctx)
        k.attention(self.ctx, self.qv, self.kc, self.vc, pos + 1, self.n_heads,
                    self.head_dim, self.groups, self.max_ctx,
                    1.0 / float(np.sqrt(self.head_dim)), self.part_acc, self.part_ml)
        k.mv_bf16(self.acc, self.o_proj, self.ctx, self.hidden,
                  self.n_heads * self.head_dim)
        k.add_(self.x, self.acc, self.hidden)

        # ---- layers.1: MoE
        k.norm(self.normed, self.x, self.n1_norm, self.hidden, self.eps)
        self.last_idx, self.last_w = self._moe()
        k.add_(self.x, self.moe_out, self.hidden)

        # ---- final norm + BACKBONE lm head (full, or the K2 active vocabulary)
        k.norm(self.normed, self.x, self.final_ln, self.hidden, self.eps)
        return self._head(), self.x, self.normed

    def nll_of(self, target_id: int) -> float:
        """-log p(target) from the logits left by the last `forward`."""
        cp = self.cp
        lg = self.logits
        m = cp.max(lg)
        lse = m + cp.log(cp.sum(cp.exp(lg - m)))
        return float(cp.asnumpy(lse - lg[target_id]))
