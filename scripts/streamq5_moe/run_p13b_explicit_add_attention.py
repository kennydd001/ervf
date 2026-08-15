from pathlib import Path


source_path = Path(__file__).with_name("run_p13a_exact_virtual_attention.py")
source = source_path.read_text(encoding="utf-8")
replacements = {
    "P13A_EXACT_VIRTUAL_ATTENTION_PREREGISTRATION.md": "P13B_EXPLICIT_ADD_ATTENTION_PREREGISTRATION.md",
    "p13a_exact_virtual_attention.json": "p13b_explicit_add_attention.json",
    "streamq5_moe_p13a_exact_virtual_attention": "streamq5_moe_p13b_explicit_add_attention",
    '''    partial[0] += partial[2];
    partial[1] += partial[3];
    float value = partial[0] + partial[1];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffU, value, offset, 32);''':
    '''    partial[0] = __fadd_rn(partial[0], partial[2]);
    partial[1] = __fadd_rn(partial[1], partial[3]);
    float value = __fadd_rn(partial[0], partial[1]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffffU, value, offset, 32);
        value = __fadd_rn(value, other);
    }''',
}
for old, new in replacements.items():
    if old not in source: raise RuntimeError(f"P13B transform target missing: {old[:80]}")
    source = source.replace(old, new)
exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": __file__})
