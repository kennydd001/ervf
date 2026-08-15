from pathlib import Path

source_path = Path(__file__).with_name("capture_p3a_routes.py")
source = source_path.read_text(encoding="utf-8")
replacements = {
    "P3A_INTEGRATED_EXPERT_DATAPLANE_PREREGISTRATION.md": "P4D_PACKED_METADATA_OVERLAP_PREREGISTRATION.md",
    "p3a_route_input_lock.json": "p4d_route_input_lock.json",
    "p3a_route_evaluator_lock.json": "p4d_route_evaluator_lock.json",
    "p3a_fresh_route_input_ids.safetensors": "p4d_fresh_route_input_ids.safetensors",
    "p3a_routes": "p4d_routes",
    "p3a_route_layers": "p4d_route_layers",
    "p3a_route_capture_result.json": "p4d_route_capture_result.json",
    "P3A_ROUTE_CAPTURE.md": "P4D_ROUTE_CAPTURE.md",
    "streamq5_moe_p3a": "streamq5_moe_p4d",
    "P3A route": "P4D route",
    "P3A - fresh": "P4D - fresh",
}
for old, new in replacements.items():
    source = source.replace(old, new)
exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": __file__})
