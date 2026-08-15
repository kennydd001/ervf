from pathlib import Path

source_path = Path(__file__).with_name("lock_p3a_route_inputs.py")
source = source_path.read_text(encoding="utf-8")
replacements = {
    "P3A_INTEGRATED_EXPERT_DATAPLANE_PREREGISTRATION.md": "P4D_PACKED_METADATA_OVERLAP_PREREGISTRATION.md",
    "p3a_fresh_route_input_ids.safetensors": "p4d_fresh_route_input_ids.safetensors",
    "p3a_route_input_lock.json": "p4d_route_input_lock.json",
    "offset = 360_000": "offset = 380_000",
    "offset = 280_000": "offset = 300_000",
    "offset = 750_000": "offset = 800_000",
    "offset = 1_050_000": "offset = 1_100_000",
    "offset = 26_600": "offset = 26_800",
    "streamq5_moe_p3a": "streamq5_moe_p4d",
    "P3A route": "P4D route",
}
for old, new in replacements.items():
    source = source.replace(old, new)
exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": __file__})
