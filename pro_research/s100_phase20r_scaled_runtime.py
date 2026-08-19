from __future__ import annotations

import types


def discover_attention_scales(rt):
    out = {}
    idx = rt.index
    for layer in rt.attn_layers:
        layer = int(layer)
        prefix = f"backbone.layers.{layer}.mixer."
        kn = [n for n in idx.entries if n.startswith(prefix) and n.endswith("k_scale")]
        vn = [n for n in idx.entries if n.startswith(prefix) and n.endswith("v_scale")]
        if len(kn) != 1 or len(vn) != 1:
            raise RuntimeError(
                f"attention layer {layer}: expected one k_scale/v_scale, got {kn}/{vn}"
            )
        ks = float(idx.get_scalar(kn[0]))
        vs = float(idx.get_scalar(vn[0]))
        out[layer] = {
            "k_name": kn[0], "v_name": vn[0],
            "k_scale": ks, "v_scale": vs,
        }
    return out


def install_scale_aware_attention(rt):
    """Install the algebraically scale-aware eager attention path.

    Cache bytes store FP8(K/k_scale) and FP8(V/v_scale). Existing attention
    kernels decode those bytes as unit-scale floats. Multiplying Q by k_scale
    restores QK; multiplying the context by v_scale restores Score*V.
    """
    if getattr(rt, "graph_mode", False):
        raise RuntimeError("Phase20R scale-aware parity requires eager graph_mode=False")

    scales = discover_attention_scales(rt)
    for layer, rec in scales.items():
        rt.layer[layer]["k_scale"] = rec["k_scale"]
        rt.layer[layer]["v_scale"] = rec["v_scale"]

    original = rt._attention

    def scaled_attention(self, i, out):
        cp, k, d = self.cp, self.k, self.layer[int(i)]
        # Preserve the exact current Q/K/V/O projection primitives.
        k.mv_bf16(self.qv, d["q_proj"], self.normed,
                   self.n_heads * self.head_dim, self.hidden)
        k.mv_bf16(self.kv_, d["k_proj"], self.normed,
                   self.kv_dim, self.hidden)
        k.mv_bf16(self.vv, d["v_proj"], self.normed,
                   self.kv_dim, self.hidden)

        scale = 1.0 / float(self.head_dim ** 0.5)
        t = self.pos + 1
        if not self.fp8_kv:
            # The checkpoint KV scales apply only to FP8 cache storage.
            k.kv_write(self.kc[i], self.kv_, self.pos,
                       self.n_kv, self.head_dim, self.max_ctx)
            k.kv_write(self.vc[i], self.vv, self.pos,
                       self.n_kv, self.head_dim, self.max_ctx)
            k.attention(self.ctx, self.qv, self.kc[i], self.vc[i], t,
                        self.n_heads, self.head_dim, self.groups,
                        self.max_ctx, scale, self.part_acc, self.part_ml)
        else:
            ks = float(d["k_scale"])
            vs = float(d["v_scale"])
            # q*(K/ks)*ks == q*K, apart from the intended FP8 cache rounding.
            self.qv *= ks
            self.kv_ *= (1.0 / ks)
            self.vv *= (1.0 / vs)
            k.kv_write_fp8(self.kc[i], self.kv_, self.pos,
                           self.n_kv, self.head_dim, self.max_ctx)
            k.kv_write_fp8(self.vc[i], self.vv, self.pos,
                           self.n_kv, self.head_dim, self.max_ctx)
            self.attn(self.ctx, self.qv, self.kc[i], self.vc[i], t,
                      self.n_heads, self.head_dim, self.groups,
                      self.max_ctx, scale, self.part_acc, self.part_ml)
            self.ctx *= vs

        k.mv_bf16(out, d["o_proj"], self.ctx, self.hidden,
                  self.n_heads * self.head_dim)

    rt._attention = types.MethodType(scaled_attention, rt)
    return scales, original
