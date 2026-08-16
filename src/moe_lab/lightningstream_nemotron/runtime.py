"""LIGHTNINGSTREAM decode runtime: N5 resident shell + N4-R2 streamed experts.

Joins the two halves the line built separately -- a correct full-depth graph
(validated on CPU in N6-A) and a fast, memory-feasible routed dataplane
(N4-R2 + N5) -- into a single-token GPU decode loop.

Resident on device: trunk, shared experts, routers, norms, embeddings, LM head,
Mamba state, conv state, KV cache, expert staging.
Host-resident: only the routed NVFP4 experts, pinned per layer.
"""

from __future__ import annotations

import numpy as np

from .fused_nvfp4 import FusedNVFP4
from .gpu_kernels import GPUKernels
from .loader import ShardIndex


def cp_asnumpy(cp, arr):
    return cp.asnumpy(arr)

CODE_BYTES = 4_988_928
SCALE_BYTES = 623_616
HALF_CODE = CODE_BYTES // 2
HALF_SCALE = SCALE_BYTES // 2
UP_CODE = HALF_CODE            # up_proj codes, row-major (2688 rows x 928 B)
UP_SCALE = HALF_SCALE          # up_proj scales, row-major (2688 x 116 B)
DOWN_PANEL_BYTES = HALF_CODE + HALF_SCALE  # down_proj panel-major (116 x 24,192 B)
PANEL_STRIDE = 24_192          # 2688 scale bytes + 16 columns x 1344 B codes


def down_panel_major(codes_raw: np.ndarray, scales_raw: np.ndarray,
                     rows: int = 2688, inter: int = 1856) -> np.ndarray:
    """Repack a row-major NVFP4 down_proj into panel-major layout.

    Panel p covers input columns 16p..16p+15. Per panel: first the `rows`
    scale bytes (one per output row, shared by the panel's columns), then the
    16 columns, each `rows` nibbles = rows//2 bytes contiguous. Pure byte
    permutation of the checkpoint values; see G-S5-C2 for the exactness check.
    """
    dc = codes_raw.reshape(rows, inter // 2)
    nib = np.empty((rows, inter), dtype=np.uint8)
    nib[:, 0::2] = dc & 15
    nib[:, 1::2] = dc >> 4
    panels = nib.reshape(rows, inter // 16, 16).transpose(1, 2, 0)
    packed = panels[..., 0::2] | (panels[..., 1::2] << 4)      # (116, 16, rows//2)
    block = np.empty((inter // 16, PANEL_STRIDE), dtype=np.uint8)
    block[:, :rows] = scales_raw.reshape(rows, inter // 16).T
    block[:, rows:] = packed.reshape(inter // 16, 16 * (rows // 2))
    return block


class LightningRuntime:
    # W1: opt-in host-path optimisation. Default off so every earlier
    # measurement still describes the path it measured.
    fast_host = False
    # D1: accumulate the routed experts in ROUTE order instead of
    # hit-then-miss order. The hit-first schedule is a latency win but it
    # makes the accumulation order depend on the LRU state, so two runs with
    # different cache history sum in a different order and float addition is
    # not associative. NERVF-5 caught this: two arms with identical
    # configuration diverged over 512 tokens.
    #
    # ADOPTED as default 2026-08-15 by phase A1. A1 changed the cache capacity
    # (72 vs 56), which changes the hit/miss pattern on nearly every layer of
    # every token: without D1 the output diverged, with D1 it was identical over
    # 2 x 256 tokens. The control arm is what gives that test its power.
    deterministic_accum = True
    # E1 fase 2.1: opt-in device-resident routing + device-LRU cache. The MoE
    # layer then runs without a single device->host synchronisation: routing,
    # slot assignment and miss staging are all kernels. Bit-comparable with
    # the host-driven path (same bytes, same order, same values); the only
    # tolerated difference source is the 6-element weight-sum association, so
    # parity is judged on TOKEN ids, exactly like A1. Default off until an
    # adoption phase says otherwise.
    device_cache = False
    _bad_pick = 0  # control-arm sabotage for E1F21-CTL; must stay 0 otherwise
    # E1 fase 2.2: opt-in graph-replay of the whole token. Requires
    # device_cache=True (the sync-free MoE path) and setup_graph() before use;
    # step() keeps working exactly as before while this is off.
    graph_mode = False

    def __init__(self, model_dir, contexts_max: int = 4096, verbose: bool = True,
                 embed_on_host: bool = False, fp8_kv: bool = True):
        import cupy as cp

        self.cp = cp
        self.embed_on_host = embed_on_host
        self.fp8_kv = fp8_kv
        self.index = ShardIndex(model_dir)
        self.cfg = self.index.config
        self.verbose = verbose
        self.max_ctx = contexts_max

        c = self.cfg
        self.pattern = self.index.pattern_string()
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
        self.m_heads = c["mamba_num_heads"]
        self.m_hdim = c["mamba_head_dim"]
        self.n_state = c["ssm_state_size"]
        self.n_groups = c["n_groups"]
        self.conv_k = c["conv_kernel"]
        self.d_inner = self.m_heads * self.m_hdim
        self.conv_dim = self.d_inner + 2 * self.n_groups * self.n_state
        self.hpg = self.m_heads // self.n_groups

        self.k = GPUKernels()
        # A1 adoption: the attention kernel is selected here, not at the call
        # site, so an A/B swaps one attribute. v1 is self.k.attention_fp8_gqa.
        self.attn = self.k.attention_fp8_gqa4
        self.fused = FusedNVFP4()

        self.moe_layers = [i for i, ch in enumerate(self.pattern) if ch == "E"]
        self.attn_layers = [i for i, ch in enumerate(self.pattern) if ch == "*"]
        self.mamba_layers = [i for i, ch in enumerate(self.pattern) if ch == "M"]

        self._load_shell()
        self._alloc_state()

    # ------------------------------------------------------------------ load
    def _u16(self, name):
        return self.cp.asarray(self.index.read_raw(name).view(np.uint16))

    def _f32(self, name):
        return self.cp.asarray(self.index.get_float32(name).astype(np.float32).ravel())

    def _u8(self, name):
        return self.cp.asarray(self.index.read_raw(name))

    def _load_shell(self):
        cp = self.cp
        idx = self.index
        self.layer = {}
        for i, ch in enumerate(self.pattern):
            p = f"backbone.layers.{i}"
            d = {"kind": ch, "norm": self._u16(f"{p}.norm.weight")}
            if ch == "M":
                m = f"{p}.mixer"
                d["in_k"] = idx.quant_kind(f"{m}.in_proj")
                d["in_q"] = d["in_k"] == "nvfp4"
                if d["in_k"] == "nvfp4":
                    d["in_codes"] = self._u8(f"{m}.in_proj.weight")
                    d["in_scales"] = self._u8(f"{m}.in_proj.weight_scale")
                    d["in_g"] = idx.get_scalar(f"{m}.in_proj.weight_scale_2")
                elif d["in_k"] == "fp8_tensor":
                    d["in_w8"] = self._u8(f"{m}.in_proj.weight")
                    d["in_s"] = idx.get_scalar(f"{m}.in_proj.weight_scale")
                else:
                    d["in_w"] = self._u16(f"{m}.in_proj.weight")
                d["out_k"] = idx.quant_kind(f"{m}.out_proj")
                d["out_q"] = d["out_k"] == "nvfp4"
                if d["out_k"] == "nvfp4":
                    d["out_codes"] = self._u8(f"{m}.out_proj.weight")
                    d["out_scales"] = self._u8(f"{m}.out_proj.weight_scale")
                    d["out_g"] = idx.get_scalar(f"{m}.out_proj.weight_scale_2")
                elif d["out_k"] == "fp8_tensor":
                    d["out_w8"] = self._u8(f"{m}.out_proj.weight")
                    d["out_s"] = idx.get_scalar(f"{m}.out_proj.weight_scale")
                else:
                    d["out_w"] = self._u16(f"{m}.out_proj.weight")
                d["conv_w"] = self._u16(f"{m}.conv1d.weight")
                d["conv_b"] = self._u16(f"{m}.conv1d.bias")
                d["A_log"] = self._f32(f"{m}.A_log")
                d["D"] = self._u16(f"{m}.D")
                d["dt_bias"] = self._u16(f"{m}.dt_bias")
                d["m_norm"] = self._u16(f"{m}.norm.weight")
            elif ch == "*":
                m = f"{p}.mixer"
                for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    d[n] = self._u16(f"{m}.{n}.weight")
            else:
                m = f"{p}.mixer"
                d["gate_w"] = self._f32(f"{m}.gate.weight")
                d["gate_b"] = self._f32(f"{m}.gate.e_score_correction_bias")
                sp = f"{m}.shared_experts"
                d["sh_up_c"] = self._u8(f"{sp}.up_proj.weight")
                d["sh_up_s"] = self._u8(f"{sp}.up_proj.weight_scale")
                d["sh_up_g"] = idx.get_scalar(f"{sp}.up_proj.weight_scale_2")
                d["sh_dn_c"] = self._u8(f"{sp}.down_proj.weight")
                d["sh_dn_s"] = self._u8(f"{sp}.down_proj.weight_scale")
                d["sh_dn_g"] = idx.get_scalar(f"{sp}.down_proj.weight_scale_2")
            self.layer[i] = d

        if self.embed_on_host:
            # A token touches 5,376 B of a 704,643,072 B table, so device
            # residency buys nothing while costing 0.656 GiB of cache slots.
            # N5 measured this placement as variant B.
            self.embed_host = idx.read_raw("backbone.embeddings.weight").view(np.uint16)
            self.embed = None
        else:
            self.embed = self._u16("backbone.embeddings.weight")
        self.norm_f = self._u16("backbone.norm_f.weight")
        self.lm_head_kind = idx.quant_kind("lm_head")
        if self.lm_head_kind == "nvfp4":
            # 3.5 Lightning ships lm_head as NVFP4: 198 MB instead of 704 MB.
            self.lm_head_codes = self._u8("lm_head.weight")
            self.lm_head_scales = self._u8("lm_head.weight_scale")
            self.lm_head_g = idx.get_scalar("lm_head.weight_scale_2")
            self.lm_head = None
        else:
            self.lm_head = self._u16("lm_head.weight")
        self.vocab = self.cfg["vocab_size"]

    def load_routed_bank(self, layers=None):
        """Pin the routed experts per layer on the host. Nothing is uploaded.

        S5 layout: up_proj stays row-major; down_proj is repacked panel-major
        (see down_panel_major) so the masked GEMV can read single columns
        straight from mapped host memory without touching zero columns.
        """
        cp = self.cp
        layers = layers or self.moe_layers
        self.bank = {}
        for layer in layers:
            n = self.n_experts
            uc = cp.cuda.alloc_pinned_memory(n * UP_CODE)
            us = cp.cuda.alloc_pinned_memory(n * UP_SCALE)
            dp = cp.cuda.alloc_pinned_memory(n * DOWN_PANEL_BYTES)
            up_codes = np.frombuffer(uc, dtype=np.uint8, count=n * UP_CODE)
            up_scales = np.frombuffer(us, dtype=np.uint8, count=n * UP_SCALE)
            down_pm = np.frombuffer(dp, dtype=np.uint8, count=n * DOWN_PANEL_BYTES)
            g = np.zeros((n, 2), dtype=np.float32)
            for e in range(n):
                pre = f"backbone.layers.{layer}.mixer.experts.{e}"
                up_codes[e * UP_CODE:(e + 1) * UP_CODE] = self.index.read_raw(
                    f"{pre}.up_proj.weight")
                up_scales[e * UP_SCALE:(e + 1) * UP_SCALE] = self.index.read_raw(
                    f"{pre}.up_proj.weight_scale")
                down_pm[e * DOWN_PANEL_BYTES:(e + 1) * DOWN_PANEL_BYTES] = \
                    down_panel_major(
                        self.index.read_raw(f"{pre}.down_proj.weight"),
                        self.index.read_raw(f"{pre}.down_proj.weight_scale")
                    ).reshape(-1)
                g[e, 0] = self.index.get_scalar(f"{pre}.down_proj.weight_scale_2")
                g[e, 1] = self.index.get_scalar(f"{pre}.up_proj.weight_scale_2")
            base = down_pm.ctypes.data
            self.bank[layer] = {
                "up_codes": up_codes, "up_scales": up_scales,
                "down_pm": down_pm,
                "down_base_ptr": base,
                "globals": g, "_pin": (uc, us, dp),
                # W1: everything the per-expert host path would otherwise rebuild
                # on every call. Views and Python floats, made once.
                "up_code_view": [up_codes[e * UP_CODE:(e + 1) * UP_CODE]
                                 for e in range(n)],
                "up_scale_view": [up_scales[e * UP_SCALE:(e + 1) * UP_SCALE]
                                  for e in range(n)],
                "down_pm_view": [down_pm[e * DOWN_PANEL_BYTES:(e + 1) * DOWN_PANEL_BYTES]
                                 for e in range(n)],
                "down_ptr": [base + e * DOWN_PANEL_BYTES for e in range(n)],
                "g_dn": [float(g[e, 0]) for e in range(n)],
                "g_up": [float(g[e, 1]) for e in range(n)],
            }
            if self.verbose:
                print(f"    pinned layer {layer:>2}: {n} experts", flush=True)

    # ----------------------------------------------------------------- state
    def _alloc_state(self):
        cp = self.cp
        self.ssm = {i: cp.zeros(self.m_heads * self.m_hdim * self.n_state, dtype=cp.float32)
                    for i in self.mamba_layers}
        self.conv = {i: cp.zeros(self.conv_dim * self.conv_k, dtype=cp.float32)
                     for i in self.mamba_layers}
        kv_dim = self.n_kv * self.head_dim
        # FP8 E4M3 KV: the checkpoint declares kv_cache_quant_algo FP8 and N3
        # measured the round trip at rel_l2 2.454e-03. Four times less attention
        # read traffic than fp32, and the freed VRAM becomes cache slots.
        kv_dtype = cp.uint8 if self.fp8_kv else cp.float32
        self.kc = {i: cp.zeros(self.max_ctx * kv_dim, dtype=kv_dtype) for i in self.attn_layers}
        self.vc = {i: cp.zeros(self.max_ctx * kv_dim, dtype=kv_dtype) for i in self.attn_layers}
        self.kv_dim = kv_dim
        self.pos = 0

        self.h = cp.zeros(self.hidden, dtype=cp.float32)
        self.tmp = cp.zeros(self.hidden, dtype=cp.float32)
        self.acc = cp.zeros(self.hidden, dtype=cp.float32)
        self.normed = cp.zeros(self.hidden, dtype=cp.float32)
        self.act = cp.zeros(max(self.moe_inter, self.shared_inter), dtype=cp.float32)
        # W1: fixed views, so the hot path does not re-slice `act` per call.
        self._act_moe = self.act[:self.moe_inter]
        self._act_shared = self.act[:self.shared_inter]
        self.proj = cp.zeros(self.d_inner + self.conv_dim + self.m_heads, dtype=cp.float32)
        self.convo = cp.zeros(self.conv_dim, dtype=cp.float32)
        self.dt = cp.zeros(self.m_heads, dtype=cp.float32)
        self.y = cp.zeros(self.d_inner, dtype=cp.float32)
        self.gn = cp.zeros(self.d_inner, dtype=cp.float32)
        self.qv = cp.zeros(self.n_heads * self.head_dim, dtype=cp.float32)
        self.kv_ = cp.zeros(kv_dim, dtype=cp.float32)
        self.vv = cp.zeros(kv_dim, dtype=cp.float32)
        self.ctx = cp.zeros(self.n_heads * self.head_dim, dtype=cp.float32)
        self.logits = cp.zeros(self.vocab, dtype=cp.float32)
        self.rlog = cp.zeros(self.n_experts, dtype=cp.float32)
        self.route_pack = cp.zeros(self.top_k * 2, dtype=cp.float32)
        # S5: staging and cache slots hold up_proj only. down_proj is never
        # copied H2D; the masked GEMV reads its nonzero columns straight from
        # the mapped pinned bank.
        self.stage_c = cp.zeros(self.top_k * UP_CODE, dtype=cp.uint8)
        self.stage_s = cp.zeros(self.top_k * UP_SCALE, dtype=cp.uint8)
        self.mstate = self.fused.alloc_masked_state(self.hidden, self.moe_inter)
        # D1: one buffer per route slot, decoupling compute order from
        # accumulation order.
        self.contrib = cp.zeros(self.top_k * self.hidden, dtype=cp.float32)
        self.copy_stream = cp.cuda.Stream(non_blocking=True)
        self.evt = [cp.cuda.Event(block=False, disable_timing=True)
                    for _ in range(self.top_k)]
        # Flash-decoding scratch for the split attention path.
        # Warp-per-position emits 4 partials per split block.
        splits = self.k.MAX_SPLITS * 4
        self.part_acc = cp.zeros(self.n_heads * splits * self.head_dim, dtype=cp.float32)
        self.part_ml = cp.zeros(self.n_heads * splits * 2, dtype=cp.float32)

    def enable_cache(self, capacity_per_layer: int, mode: str = "up_only"):
        """Per-layer LRU of resident expert records.

        Sized from N7-A's measured locality: 2.011 of 6 experts are shared
        between consecutive tokens, so an LRU earns its slots even though N6-A
        showed a shallow global popularity spread. Those are different
        properties and only the temporal one matters here.

        ``mode`` decides what a slot holds, at exactly 2x the bytes per slot:

        up_only : up_proj codes + scales (2,806,272 B). down_proj is fetched
                  from mapped host on EVERY call, hit or miss (S5).
        full    : the same plus the panel-major down_proj record, so a hit needs
                  no PCIe at all and skips gather_down_sparse entirely.

        S11 compares the two at equal cache BYTES, i.e. half the capacity for
        ``full``. Default is unchanged so every earlier measurement still
        describes the path it measured.
        """
        from collections import OrderedDict

        if mode not in ("up_only", "full"):
            raise ValueError(f"cache mode must be up_only or full, got {mode!r}")
        cp = self.cp
        self.cache_mode = mode
        self.cache = {}
        # E1 fase 2.1: the device-LRU tables are sized by capacity and carry
        # live routing state; rebuild them with the host cache or a capacity
        # change would run new semantics over stale slots.
        self._dev_cache = {}
        # E1 fase 2.2: a captured graph binds the OLD cache buffers and device
        # tables by pointer; changing the cache invalidates it.
        self._graph = None
        for layer in self.moe_layers:
            entry = {
                "codes": cp.zeros(capacity_per_layer * UP_CODE, dtype=cp.uint8),
                "scales": cp.zeros(capacity_per_layer * UP_SCALE, dtype=cp.uint8),
                "map": OrderedDict(),
                "cap": capacity_per_layer,
            }
            # W1: one view per slot, made once, so the hot path indexes a list
            # instead of building a fresh cupy ndarray on every expert call.
            entry["slot_codes"] = [entry["codes"][k * UP_CODE:(k + 1) * UP_CODE]
                                   for k in range(capacity_per_layer)]
            entry["slot_scales"] = [entry["scales"][k * UP_SCALE:(k + 1) * UP_SCALE]
                                    for k in range(capacity_per_layer)]
            if mode == "full":
                entry["down"] = cp.zeros(capacity_per_layer * DOWN_PANEL_BYTES,
                                         dtype=cp.uint8)
                entry["down_ptr"] = int(entry["down"].data.ptr)
                entry["slot_down"] = [
                    entry["down"][k * DOWN_PANEL_BYTES:(k + 1) * DOWN_PANEL_BYTES]
                    for k in range(capacity_per_layer)]
            self.cache[layer] = entry
        self.cache_stats = {"hits": 0, "misses": 0}
        return sum(c["codes"].nbytes + c["scales"].nbytes
                   + (c["down"].nbytes if mode == "full" else 0)
                   for c in self.cache.values())

    def reset(self):
        for d in (self.ssm, self.conv):
            for v in d.values():
                v.fill(0)
        self.pos = 0
        # E1 fase 2.2: the graph reads pos from a device buffer; reset it too.
        # The sync guards ordering against launches on the graph stream.
        if hasattr(self, "_pos_dev"):
            self._pos_dev.fill(0)
            self.cp.cuda.Device(0).synchronize()

    # ------------------------------------------------------------ sub-blocks
    def _mamba(self, i, out):
        cp, k, d = self.cp, self.k, self.layer[i]
        if d["in_k"] == "nvfp4":
            self.fused.gemv_into(self.proj, d["in_codes"], d["in_scales"], self.normed,
                                 d["in_g"], self.proj.size, self.hidden)
        elif d["in_k"] == "fp8_tensor":
            k.mv_fp8_tensor(self.proj, d["in_w8"], self.normed, d["in_s"],
                            self.proj.size, self.hidden)
        else:
            k.mv_bf16(self.proj, d["in_w"], self.normed, self.proj.size, self.hidden)

        z = self.proj[:self.d_inner]
        xbc = self.proj[self.d_inner:self.d_inner + self.conv_dim]
        dtr = self.proj[self.d_inner + self.conv_dim:]

        k.conv_step(self.convo, self.conv[i], xbc, d["conv_w"], d["conv_b"],
                    self.conv_dim, self.conv_k)
        x = self.convo[:self.d_inner]
        Bv = self.convo[self.d_inner:self.d_inner + self.n_groups * self.n_state]
        Cv = self.convo[self.d_inner + self.n_groups * self.n_state:]

        k.dt_activate(self.dt, dtr, d["dt_bias"], self.m_heads, 0.0, 3.4e38)
        k.ssm_step(self.y, self.ssm[i], x, Bv, Cv, self.dt, d["A_log"], d["D"],
                   self.m_heads, self.m_hdim, self.n_state, self.hpg)
        k.gated_norm(self.gn, self.y, z, d["m_norm"], self.d_inner,
                     self.d_inner // self.n_groups, self.eps)

        if d["out_k"] == "nvfp4":
            self.fused.gemv_into(out, d["out_codes"], d["out_scales"], self.gn,
                                 d["out_g"], self.hidden, self.d_inner)
        elif d["out_k"] == "fp8_tensor":
            k.mv_fp8_tensor(out, d["out_w8"], self.gn, d["out_s"],
                            self.hidden, self.d_inner)
        else:
            k.mv_bf16(out, d["out_w"], self.gn, self.hidden, self.d_inner)

    def _attention(self, i, out):
        cp, k, d = self.cp, self.k, self.layer[i]
        k.mv_bf16(self.qv, d["q_proj"], self.normed, self.n_heads * self.head_dim, self.hidden)
        k.mv_bf16(self.kv_, d["k_proj"], self.normed, self.kv_dim, self.hidden)
        k.mv_bf16(self.vv, d["v_proj"], self.normed, self.kv_dim, self.hidden)

        scale = 1.0 / float(np.sqrt(self.head_dim))
        if self.fp8_kv and self.graph_mode:
            # E1 fase 2.2: pos/t live on device; grids are fixed so the whole
            # call is capturable. Bit-comparable with the eager v4 path.
            k.kv_write_fp8_dp(self.kc[i], self.kv_, self._pos_dev, self.n_kv,
                              self.head_dim, self.max_ctx)
            k.kv_write_fp8_dp(self.vc[i], self.vv, self._pos_dev, self.n_kv,
                              self.head_dim, self.max_ctx)
            k.attention_fp8_gqa4_dp(self.ctx, self.qv, self.kc[i], self.vc[i],
                                    self._pos_dev, self.n_heads, self.head_dim,
                                    self.groups, self.max_ctx, scale,
                                    self.part_acc, self.part_ml)
            k.mv_bf16(out, d["o_proj"], self.ctx, self.hidden,
                      self.n_heads * self.head_dim)
            return
        t = self.pos + 1
        if self.fp8_kv:
            k.kv_write_fp8(self.kc[i], self.kv_, self.pos, self.n_kv,
                           self.head_dim, self.max_ctx)
            k.kv_write_fp8(self.vc[i], self.vv, self.pos, self.n_kv,
                           self.head_dim, self.max_ctx)
            # A1 adoption: dispatch through self.attn, which defaults to the v4
            # kernel (E4: bitwise exact, -17.8%). Scripts written BEFORE the
            # adoption swap kernels by assigning rt.k.attention_fp8_gqa; that
            # path is no longer called, so such an A/B collapses to a zero-gain
            # null result instead of silently mismeasuring. Swap rt.attn instead.
            self.attn(self.ctx, self.qv, self.kc[i], self.vc[i], t,
                      self.n_heads, self.head_dim, self.groups,
                      self.max_ctx, scale, self.part_acc, self.part_ml)
        else:
            k.kv_write(self.kc[i], self.kv_, self.pos, self.n_kv,
                       self.head_dim, self.max_ctx)
            k.kv_write(self.vc[i], self.vv, self.pos, self.n_kv,
                       self.head_dim, self.max_ctx)
            k.attention(self.ctx, self.qv, self.kc[i], self.vc[i], t,
                        self.n_heads, self.head_dim, self.groups, self.max_ctx,
                        scale, self.part_acc, self.part_ml)
        k.mv_bf16(out, d["o_proj"], self.ctx, self.hidden, self.n_heads * self.head_dim)

    def _route_device(self, i):
        """Router entirely on device; returns a single packed array to read back.

        The expert ids must reach the host to index the pinned bank, but each
        device->host transfer costs a full synchronisation. Two transfers per
        layer over 23 layers was 46 syncs per token and measured 0.339 ms per
        layer for a 344 kFLOP GEMV -- almost all of it latency, not arithmetic.
        Packing ids and weights into one buffer halves that immediately.
        """
        cp, d = self.cp, self.layer[i]
        self.k.mv_f32(self.rlog, d["gate_w"], self.normed, self.n_experts, self.hidden)
        scores = 1.0 / (1.0 + cp.exp(-self.rlog))
        choice = scores + d["gate_b"]
        idx = cp.argsort(-choice)[: self.top_k]
        w = scores[idx]
        w = w / (w.sum() + 1e-20) * self.scaling
        self.route_pack[: self.top_k] = idx.astype(cp.float32)
        self.route_pack[self.top_k:] = w
        return self.route_pack

    def _route(self, i):
        packed = cp_asnumpy(self.cp, self._route_device(i))
        return packed[: self.top_k].astype(int), packed[self.top_k:].astype(np.float64)

    def _moe_cached(self, i, out):
        """MoE with a per-layer LRU: only misses cross PCIe."""
        cp, d = self.cp, self.layer[i]

        # Router runs on device, then the SHARED expert -- which does not depend
        # on routing at all -- is launched before the route readback. The
        # unavoidable device->host sync for the expert ids then overlaps with
        # real work instead of stalling the pipeline.
        packed = self._route_device(i)
        out.fill(0)
        self.fused.gemv_into(self.act[:self.shared_inter], d["sh_up_c"], d["sh_up_s"],
                             self.normed, d["sh_up_g"], self.shared_inter, self.hidden,
                             apply_relu2=True)
        self.fused.gemv_into(out, d["sh_dn_c"], d["sh_dn_s"],
                             self.act[:self.shared_inter], d["sh_dn_g"],
                             self.hidden, self.shared_inter)

        host = cp_asnumpy(cp, packed)
        idx = host[: self.top_k].astype(int)
        w = host[self.top_k:].astype(np.float64)

        bank, c = self.bank[i], self.cache[i]
        cmap, cap = c["map"], c["cap"]
        full = getattr(self, "cache_mode", "up_only") == "full"
        det = self.deterministic_accum

        # Issue miss transfers (up halves only), then compute HITS FIRST while
        # they land. Only a miss waits on its own event; a global synchronize
        # would make every hit pay for the slowest miss.
        slots, needs_wait = [], []
        with self.copy_stream:
            for s, e in enumerate(idx):
                e = int(e)
                if e in cmap:
                    cmap.move_to_end(e)
                    slots.append(cmap[e])
                    needs_wait.append(False)
                    self.cache_stats["hits"] += 1
                    continue
                self.cache_stats["misses"] += 1
                if len(cmap) < cap:
                    slot = len(cmap)
                else:
                    _, slot = cmap.popitem(last=False)
                cmap[e] = slot
                slots.append(slot)
                needs_wait.append(True)
                c["codes"][slot * UP_CODE:(slot + 1) * UP_CODE].set(
                    bank["up_codes"][e * UP_CODE:(e + 1) * UP_CODE],
                    stream=self.copy_stream)
                c["scales"][slot * UP_SCALE:(slot + 1) * UP_SCALE].set(
                    bank["up_scales"][e * UP_SCALE:(e + 1) * UP_SCALE],
                    stream=self.copy_stream)
                if full:
                    c["down"][slot * DOWN_PANEL_BYTES:(slot + 1) * DOWN_PANEL_BYTES].set(
                        bank["down_pm"][e * DOWN_PANEL_BYTES:(e + 1) * DOWN_PANEL_BYTES],
                        stream=self.copy_stream)
                self.evt[s].record(self.copy_stream)

        order = [s for s in range(len(idx)) if not needs_wait[s]]
        order += [s for s in range(len(idx)) if needs_wait[s]]

        for s in order:
            e = idx[s]
            if needs_wait[s]:
                cp.cuda.get_current_stream().wait_event(self.evt[s])
            sl = slots[s]
            self.fused.gemv_into(self.act[:self.moe_inter],
                                 c["codes"][sl * UP_CODE:(sl + 1) * UP_CODE],
                                 c["scales"][sl * UP_SCALE:(sl + 1) * UP_SCALE],
                                 self.normed, float(bank["globals"][e, 1]),
                                 self.moe_inter, self.hidden, apply_relu2=True)
            # down_proj. up_only: masked columns straight from the mapped host
            # bank, so only nonzero ReLU2 columns cross PCIe (~9% of down's
            # bytes) -- but on every call, hit or miss. full: the record is
            # already resident, so the gather is skipped entirely and the same
            # masked GEMV reads device memory.
            if full:
                self.fused.down_masked_into(
                    self.tmp, c["down_ptr"] + sl * DOWN_PANEL_BYTES,
                    self.act[:self.moe_inter], self.mstate,
                    float(bank["globals"][e, 0]), self.hidden, self.moe_inter,
                    gather_from_host=False)
            else:
                self.fused.down_masked_into(
                    self.tmp, bank["down_base_ptr"] + int(e) * DOWN_PANEL_BYTES,
                    self.act[:self.moe_inter], self.mstate,
                    float(bank["globals"][e, 0]), self.hidden, self.moe_inter)
            if det:
                self.contrib[s * self.hidden:(s + 1) * self.hidden] = self.tmp
            else:
                self.fused.accumulate_into(out, self.tmp, float(w[s]), self.hidden)
        if det:
            for s in range(len(idx)):
                self.fused.accumulate_into(
                    out, self.contrib[s * self.hidden:(s + 1) * self.hidden],
                    float(w[s]), self.hidden)
        return idx, w

    def _moe_cached_fast(self, i, out):
        """W1: `_moe_cached` with the per-call host work precomputed away.

        Identical kernels, identical arguments, identical order. The only thing
        that changes is how much Python runs between the launches -- S14 measured
        4.7 ms per token of GPU idle around exactly this code.
        """
        cp, d = self.cp, self.layer[i]

        packed = self._route_device(i)
        out.fill(0)
        act_sh, act_moe = self._act_shared, self._act_moe
        fused = self.fused
        fused.gemv_into(act_sh, d["sh_up_c"], d["sh_up_s"], self.normed,
                        d["sh_up_g"], self.shared_inter, self.hidden,
                        apply_relu2=True)
        fused.gemv_into(out, d["sh_dn_c"], d["sh_dn_s"], act_sh, d["sh_dn_g"],
                        self.hidden, self.shared_inter)

        flat = cp_asnumpy(cp, packed).tolist()
        top_k = self.top_k
        idx = [int(v) for v in flat[:top_k]]
        w = flat[top_k:]

        bank, c = self.bank[i], self.cache[i]
        cmap, cap = c["map"], c["cap"]
        slot_codes, slot_scales = c["slot_codes"], c["slot_scales"]
        full = self.cache_mode == "full"
        hits = misses = 0

        # One pass builds slots, the wait flags and the hit-first order.
        slots = [0] * top_k
        wait = [False] * top_k
        order_hit, order_miss, pending = [], [], []
        for s in range(top_k):
            e = idx[s]
            slot = cmap.get(e, -1)
            if slot >= 0:
                cmap.move_to_end(e)
                slots[s] = slot
                hits += 1
                order_hit.append(s)
                continue
            misses += 1
            if len(cmap) < cap:
                slot = len(cmap)
            else:
                _, slot = cmap.popitem(last=False)
            cmap[e] = slot
            slots[s] = slot
            wait[s] = True
            order_miss.append(s)
            pending.append((s, e, slot))

        if pending:
            stream = self.copy_stream
            with stream:
                for s, e, slot in pending:
                    slot_codes[slot].set(bank["up_code_view"][e], stream=stream)
                    slot_scales[slot].set(bank["up_scale_view"][e], stream=stream)
                    if full:
                        c["slot_down"][slot].set(bank["down_pm_view"][e], stream=stream)
                    self.evt[s].record(stream)

        st = self.cache_stats
        st["hits"] += hits
        st["misses"] += misses

        g_up, g_dn, down_ptr = bank["g_up"], bank["g_dn"], bank["down_ptr"]
        cur = cp.cuda.get_current_stream()
        mstate, tmp = self.mstate, self.tmp
        hidden, inter = self.hidden, self.moe_inter
        for s in order_hit + order_miss:
            e = idx[s]
            if wait[s]:
                cur.wait_event(self.evt[s])
            sl = slots[s]
            fused.gemv_into(act_moe, slot_codes[sl], slot_scales[sl], self.normed,
                            g_up[e], inter, hidden, apply_relu2=True)
            if full:
                fused.down_masked_into(tmp, c["down_ptr"] + sl * DOWN_PANEL_BYTES,
                                       act_moe, mstate, g_dn[e], hidden, inter,
                                       gather_from_host=False)
            else:
                fused.down_masked_into(tmp, down_ptr[e], act_moe, mstate,
                                       g_dn[e], hidden, inter)
            fused.accumulate_into(out, tmp, w[s], hidden)
        return np.asarray(idx), np.asarray(w)

    def _moe_dev(self, i, out):
        """E1 fase 2.1: the MoE layer with routing, LRU assignment and miss
        staging entirely on device -- no device->host synchronisation at all.

        Stream choreography: route+assign on the main stream, then the miss
        staging forks onto the copy stream while the shared expert (which does
        not depend on routing) computes on the main stream; the expert loop
        joins on the staging event. Every kernel reads its expert id, slot,
        global scale and route weight from device buffers, so nothing here
        needs a value on the host. cache_mode must be up_only: down_proj
        columns are read from the mapped bank per call exactly as in
        _moe_cached.
        """
        cp, k, d, fused = self.cp, self.k, self.layer[i], self.fused
        bank, c = self.bank[i], self.cache[i]
        if not hasattr(self, "_dev_cache"):
            self._dev_cache = {}
        if i not in self._dev_cache:
            self._dev_cache[i] = fused.alloc_device_cache(
                self.n_experts, c["cap"], self.top_k, bank["globals"])
        dev = self._dev_cache[i]

        k.mv_f32(self.rlog, d["gate_w"], self.normed, self.n_experts, self.hidden)
        fused.route_topk(self.rlog, d["gate_b"], dev["ids"], dev["w"],
                         self.n_experts, self.top_k, self.scaling,
                         bad_pick=self._bad_pick)
        fused.cache_assign(dev, dev["ids"], c["cap"], self.top_k)
        self.evt[0].record()
        with self.copy_stream:
            self.copy_stream.wait_event(self.evt[0])
            fused.cache_fetch(bank["up_codes"].ctypes.data,
                              bank["up_scales"].ctypes.data,
                              c["codes"], c["scales"], dev,
                              UP_CODE, UP_SCALE, self.top_k)
            self.evt[1].record(self.copy_stream)

        out.fill(0)
        fused.gemv_into(self._act_shared, d["sh_up_c"], d["sh_up_s"],
                        self.normed, d["sh_up_g"], self.shared_inter,
                        self.hidden, apply_relu2=True)
        fused.gemv_into(out, d["sh_dn_c"], d["sh_dn_s"],
                        self._act_shared, d["sh_dn_g"],
                        self.hidden, self.shared_inter)

        cp.cuda.get_current_stream().wait_event(self.evt[1])
        for s in range(self.top_k):
            fused.gemv_ervf_indirect(self._act_moe, c["codes"], c["scales"],
                                     dev, s, dev["globals"], 1, self.normed,
                                     self.moe_inter, self.hidden, True,
                                     UP_CODE, UP_SCALE)
            fused.down_masked_into_indirect(
                self.contrib[s * self.hidden:(s + 1) * self.hidden],
                bank["down_base_ptr"], dev, s, dev["globals"],
                self._act_moe, self.mstate, self.hidden,
                self.moe_inter, DOWN_PANEL_BYTES)
        for s in range(self.top_k):
            fused.accumulate_indirect(
                out, self.contrib[s * self.hidden:(s + 1) * self.hidden],
                dev["w"][s:], self.hidden)
        return None, None

    def _moe(self, i, out):
        if getattr(self, "cache", None):
            if getattr(self, "device_cache", False):
                return self._moe_dev(i, out)
            if getattr(self, "fast_host", False):
                return self._moe_cached_fast(i, out)
            return self._moe_cached(i, out)
        cp, d = self.cp, self.layer[i]
        idx, w = self._route(i)
        bank = self.bank[i]

        # Intra-layer pipelining: expert s+1's up half transfers while expert s
        # computes. Cross-LAYER prefetch is causally impossible -- layer L+1's
        # route depends on layer L's output (and S1 measured route prediction
        # at recall@24 0.724, below its closure gate) -- so this is the whole
        # overlap that is available.
        def issue(s: int) -> None:
            e = idx[s]
            self.stage_c[s * UP_CODE:(s + 1) * UP_CODE].set(
                bank["up_codes"][e * UP_CODE:(e + 1) * UP_CODE], stream=self.copy_stream)
            self.stage_s[s * UP_SCALE:(s + 1) * UP_SCALE].set(
                bank["up_scales"][e * UP_SCALE:(e + 1) * UP_SCALE], stream=self.copy_stream)
            self.evt[s].record(self.copy_stream)

        with self.copy_stream:
            issue(0)

        out.fill(0)
        for s, e in enumerate(idx):
            if s + 1 < len(idx):
                with self.copy_stream:
                    issue(s + 1)
            cp.cuda.get_current_stream().wait_event(self.evt[s])
            self.fused.gemv_into(self.act[:self.moe_inter],
                                 self.stage_c[s * UP_CODE:(s + 1) * UP_CODE],
                                 self.stage_s[s * UP_SCALE:(s + 1) * UP_SCALE],
                                 self.normed, float(bank["globals"][e, 1]),
                                 self.moe_inter, self.hidden, apply_relu2=True)
            self.fused.down_masked_into(
                self.tmp, bank["down_base_ptr"] + int(e) * DOWN_PANEL_BYTES,
                self.act[:self.moe_inter], self.mstate,
                float(bank["globals"][e, 0]), self.hidden, self.moe_inter)
            self.fused.accumulate_into(out, self.tmp, float(w[s]), self.hidden)

        self.fused.gemv_into(self.act[:self.shared_inter], d["sh_up_c"], d["sh_up_s"],
                             self.normed, d["sh_up_g"], self.shared_inter, self.hidden,
                             apply_relu2=True)
        self.fused.gemv_into(self.tmp, d["sh_dn_c"], d["sh_dn_s"],
                             self.act[:self.shared_inter], d["sh_dn_g"],
                             self.hidden, self.shared_inter)
        self.k.add_(out, self.tmp, self.hidden)
        return idx, w

    # ------------------------------------------------------------------ step
    def step(self, token_id: int, capture_routes=None) -> int:
        cp, k = self.cp, self.k
        # BF16 -> FP32 is a 16-bit left shift reinterpreted as float.
        if self.embed_on_host:
            row = cp.asarray(
                self.embed_host[token_id * self.hidden:(token_id + 1) * self.hidden])
        else:
            row = self.embed[token_id * self.hidden:(token_id + 1) * self.hidden]
        self.h[:] = (row.astype(cp.uint32) << cp.uint32(16)).view(cp.float32)

        for i, ch in enumerate(self.pattern):
            d = self.layer[i]
            k.norm(self.normed, self.h, d["norm"], self.hidden, self.eps)
            if ch == "M":
                self._mamba(i, self.acc)
            elif ch == "*":
                self._attention(i, self.acc)
            else:
                idx, w = self._moe(i, self.acc)
                if capture_routes is not None and idx is not None:
                    capture_routes.setdefault(str(i), []).append(idx.tolist())
            k.add_(self.h, self.acc, self.hidden)

        k.norm(self.normed, self.h, self.norm_f, self.hidden, self.eps)
        if self.lm_head_kind == "nvfp4":
            self.fused.gemv_into(self.logits, self.lm_head_codes, self.lm_head_scales,
                                 self.normed, self.lm_head_g, self.vocab, self.hidden)
        else:
            k.mv_bf16(self.logits, self.lm_head, self.normed, self.vocab, self.hidden)
        self.pos += 1
        return int(cp.argmax(self.logits))

    # ----------------------------------------------------- E1 fase 2.2 ------
    # Graph-replay of the whole token. Built per the frozen preregistration
    # (E1F22_GRAPH_CAPTURE_PREREGISTRATION_2026-08-15.md). STATUS 2026-08-16:
    # PROVEN IN PRODUCTION by the pro_research line -- V4-V6 ran this exact
    # machinery (setup_graph/step_graph/_step_body_graph) bit-exact on the
    # real Lightning checkpoint up to the 47.41 tok/s record
    # (pro_research/results/PRO_V4_GRAPH_SELECTIVE.json). The treesweep200
    # line's own gated A/B (E1F22) was never run on the Nano checkpoint; treat
    # that specific report as superseded by the V4-V6 evidence.

    def setup_graph(self):
        """Allocate device token/pos state, pin the embedding table, capture
        one token body into a CUDA graph. Call AFTER enable_cache(); calling
        enable_cache() afterwards invalidates the graph (it binds the cache
        buffers by pointer)."""
        cp = self.cp
        if not getattr(self, "device_cache", False):
            raise RuntimeError("setup_graph requires device_cache=True "
                               "(E1 fase 2.1 sync-free MoE path)")
        if not self.cache:
            raise RuntimeError("enable_cache() must run before setup_graph()")
        if getattr(self, "_graph", None) is not None:
            return
        free0 = cp.cuda.Device(0).mem_info[0]
        self._tok_dev = cp.zeros(1, dtype=cp.int32)
        self._pos_dev = cp.zeros(1, dtype=cp.int32)
        self._am_max = cp.zeros(256, dtype=cp.float32)
        self._am_idx = cp.zeros(256, dtype=cp.int32)
        if self.embed_on_host:
            # The gather kernel dereferences the table, so it must be pinned
            # + mapped; +0.656 GiB pinned host RAM, reported by the runner.
            nbytes = self.embed_host.nbytes
            pm = cp.cuda.alloc_pinned_memory(nbytes)
            np.frombuffer(pm, dtype=np.uint8, count=nbytes)[:] = \
                self.embed_host.view(np.uint8)
            self._embed_pinned = pm
            self._embed_tbl_ptr = int(pm.ptr)
        else:
            self._embed_pinned = None
            self._embed_tbl_ptr = int(self.embed.data.ptr)
        # Staging is a 256-slot pinned ring, not one slot: a 4-byte async H2D
        # copy reads the pinned source at GPU-execution time, so a single slot
        # would be overwritten by the host before the copy runs when prompt
        # tokens are fed back-to-back (this race produced degenerate output in
        # the first multi-seq graph prototypes; V4's driver dodged it with a
        # sync per prompt token). 256 slots let the host run well ahead;
        # callers feeding >256 prompt tokens without any sync must sync.
        self._stage_mem = cp.cuda.alloc_pinned_memory(4 * 256)
        self._stage_np = np.frombuffer(self._stage_mem, dtype=np.int32)
        self._stage_i = 0
        self._ring_size = 8192
        self._ring_mem = cp.cuda.alloc_pinned_memory(4 * self._ring_size)
        self._ring_np = np.frombuffer(self._ring_mem, dtype=np.int32)
        self._ring_i = 0
        self.graph_mode = True
        s = cp.cuda.Stream(non_blocking=True)
        self._graph_stream = s
        with s:
            self._step_body_graph()  # warmup: compiles every kernel, mutates
        s.synchronize()            # state; reset() below restores it
        s.begin_capture()
        with s:
            self._step_body_graph()
        self._graph = s.end_capture()
        s.synchronize()
        self.reset()
        free1 = cp.cuda.Device(0).mem_info[0]
        self.graph_extra_vram_bytes = int(free0 - free1)

    def _step_body_graph(self):
        """One decode token with zero host reads -- the captured body. Every
        scalar a kernel needs lives on device: token id, position, routes."""
        k = self.k
        k.embed_gather(self.h, self._embed_tbl_ptr, self._tok_dev, self.hidden)
        for i, ch in enumerate(self.pattern):
            d = self.layer[i]
            k.norm(self.normed, self.h, d["norm"], self.hidden, self.eps)
            if ch == "M":
                self._mamba(i, self.acc)
            elif ch == "*":
                self._attention(i, self.acc)
            else:
                self._moe(i, self.acc)
            k.add_(self.h, self.acc, self.hidden)
        k.norm(self.normed, self.h, self.norm_f, self.hidden, self.eps)
        if self.lm_head_kind == "nvfp4":
            self.fused.gemv_into(self.logits, self.lm_head_codes,
                                 self.lm_head_scales, self.normed,
                                 self.lm_head_g, self.vocab, self.hidden)
        else:
            k.mv_bf16(self.logits, self.lm_head, self.normed, self.vocab,
                      self.hidden)
        k.argmax_logits(self._tok_dev, self.logits, self.vocab,
                        self._am_max, self._am_idx)
        k.pos_increment(self._pos_dev)

    def step_graph(self, token_id=None):
        """Replay the captured token. ``token_id`` stages a prompt token first
        (stream-ordered 4-byte H2D, no sync); None means decode: embed the id
        that argmax left in tok_dev. The id this launch produces lands in the
        pinned ring; read it with ring_harvest."""
        g = getattr(self, "_graph", None)
        if g is None:
            raise RuntimeError("no graph: call setup_graph() first, and do "
                               "not call enable_cache() after it")
        s = self._graph_stream
        rt = self.cp.cuda.runtime
        if token_id is not None:
            j = self._stage_i % 256
            self._stage_np[j] = token_id
            rt.memcpyAsync(self._tok_dev.data.ptr, self._stage_mem.ptr + 4 * j,
                           4, rt.memcpyHostToDevice, s.ptr)
            self._stage_i += 1
        g.launch(s)
        rt.memcpyAsync(self._ring_mem.ptr + 4 * self._ring_i,
                       self._tok_dev.data.ptr, 4,
                       rt.memcpyDeviceToHost, s.ptr)
        self._ring_i = (self._ring_i + 1) % self._ring_size

    def ring_harvest(self, start: int, count: int):
        """Synchronise the graph stream and read `count` ids from the ring
        starting at ring index `start`. Ring slot j holds the id produced by
        launch j, i.e. the token that launch j+1 embeds: after P prompt
        launches the first generated id is at ring index P-1."""
        self._graph_stream.synchronize()
        return [int(self._ring_np[(start + j) % self._ring_size])
                for j in range(count)]



