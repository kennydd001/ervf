from __future__ import annotations

import hashlib
import inspect
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP, Qwen3MoeSparseMoeBlock


ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH / "src"))
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_checkpoint_tensors
from scripts.streamq5_moe.run_p9b_structured_wanda_pruning import apply_structured_pruning_


MODEL = ROOT / "models/qwen3-30b-a3b-base"
P9B_MASKS = ROOT / "reports/runs/streamq5_moe/p9b_structured_wanda_keep.safetensors"
P9E_MASKS = ROOT / "reports/runs/streamq5_moe/p9e1_group_balanced_keep.safetensors"
P9B_RESULT = ROOT / "reports/streamq5_moe/p9b_structured_wanda_validation.json"
P9E_RESULT = ROOT / "reports/streamq5_moe/p9e1_group_balanced_wanda_validation.json"
P9B_SCRIPT = ROOT / "scripts/streamq5_moe/run_p9b_structured_wanda_pruning.py"
P9E_SCRIPT = ROOT / "scripts/streamq5_moe/run_p9e1_group_balanced_wanda.py"
OUTPUT = ROOT / "reports/streamq5_moe/gaugepack_p9d1_p9b_mutation_audit.json"
REPORT = ROOT / "reports/streamq5_moe/GAUGEPACK_P9D1_P9B_MUTATION_AUDIT_2026-08-12.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def corrected_pruning_(expert: Qwen3MoeMLP, keep: torch.Tensor) -> None:
    mask = torch.zeros(768, dtype=torch.bool, device=expert.down_proj.weight.device)
    mask[keep.long().to(mask.device)] = True
    expert.gate_proj.weight.masked_fill_(~mask[:, None], 0)
    expert.up_proj.weight.masked_fill_(~mask[:, None], 0)
    expert.down_proj.weight.masked_fill_(~mask[None, :], 0)


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite completed P9B mutation audit")
    started = time.perf_counter()
    required = (P9B_MASKS, P9E_MASKS, P9B_RESULT, P9E_RESULT, P9B_SCRIPT, P9E_SCRIPT)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError([str(path) for path in required if not path.is_file()])

    p9b_masks = load_file(P9B_MASKS)
    p9e_masks = load_file(P9E_MASKS)
    mask_index_mismatches = 0
    identical_expert_layer_masks = 0
    for layer in range(48):
        p9b = p9b_masks[f"layer_{layer:02d}"]
        p9e = p9e_masks[f"layer_{layer:02d}"]
        mask_index_mismatches += int((p9b != p9e).sum())
        identical_expert_layer_masks += sum(int(torch.equal(p9b[expert], p9e[expert])) for expert in range(128))

    p9b_result = json.loads(P9B_RESULT.read_text(encoding="utf-8"))
    p9e_result = json.loads(P9E_RESULT.read_text(encoding="utf-8"))
    output_equalities = {
        "candidate_context_ce_exact": p9b_result["candidate_context_ce"] == p9e_result["candidate_context_ce"],
        "candidate_ce_exact": p9b_result["candidate_ce"] == p9e_result["candidate_ce"],
        "relative_cross_entropy_exact": p9b_result["relative_cross_entropy_increase"] == p9e_result["relative_cross_entropy_increase"],
        "top1_agreement_exact": p9b_result["top1_agreement"] == p9e_result["top1_agreement"],
        "final_hidden_error_exact": p9b_result["final_hidden_error"] == p9e_result["final_hidden_error"],
        "all_48_layer_hidden_errors_exact": [row["hidden_error"] for row in p9b_result["layers"]] == [row["hidden_error"] for row in p9e_result["layers"]],
    }

    weight_map = checkpoint_weight_map(MODEL)
    base = "model.layers.0.mlp.experts.0"
    identities = {kind: f"{base}.{kind}_proj.weight" for kind in ("gate", "up", "down")}
    source = load_checkpoint_tensors(MODEL, list(identities.values()), weight_map)
    config = Qwen3MoeConfig.from_pretrained(MODEL, local_files_only=True)
    expert = Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size)
    expert.gate_proj.weight = torch.nn.Parameter(source[identities["gate"]].clone())
    expert.up_proj.weight = torch.nn.Parameter(source[identities["up"]].clone())
    expert.down_proj.weight = torch.nn.Parameter(source[identities["down"]].clone())
    expert.eval()
    wrapper = SimpleNamespace(mlp=SimpleNamespace(experts=torch.nn.ModuleList([expert])))
    keep = p9b_masks["layer_00"][0].long()
    mask = torch.zeros(768, dtype=torch.bool)
    mask[keep] = True

    before_hashes = {kind: tensor_sha(getattr(expert, f"{kind}_proj").weight) for kind in ("gate", "up", "down")}
    removed_nonzero_before = {
        "gate": int(torch.count_nonzero(expert.gate_proj.weight[~mask])),
        "up": int(torch.count_nonzero(expert.up_proj.weight[~mask])),
        "down": int(torch.count_nonzero(expert.down_proj.weight[:, ~mask])),
    }
    selection_storage_is_distinct = {
        "gate": expert.gate_proj.weight[~mask].untyped_storage().data_ptr() != expert.gate_proj.weight.untyped_storage().data_ptr(),
        "up": expert.up_proj.weight[~mask].untyped_storage().data_ptr() != expert.up_proj.weight.untyped_storage().data_ptr(),
        "down": expert.down_proj.weight[:, ~mask].untyped_storage().data_ptr() != expert.down_proj.weight.untyped_storage().data_ptr(),
    }
    generator = torch.Generator(device="cpu").manual_seed(12082026)
    probe = torch.randn((4, config.hidden_size), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    with torch.no_grad():
        output_before = expert(probe).clone()
    apply_structured_pruning_(wrapper, keep.unsqueeze(0))
    after_bug_hashes = {kind: tensor_sha(getattr(expert, f"{kind}_proj").weight) for kind in ("gate", "up", "down")}
    removed_nonzero_after_bug = {
        "gate": int(torch.count_nonzero(expert.gate_proj.weight[~mask])),
        "up": int(torch.count_nonzero(expert.up_proj.weight[~mask])),
        "down": int(torch.count_nonzero(expert.down_proj.weight[:, ~mask])),
    }
    with torch.no_grad():
        output_after_bug = expert(probe).clone()

    corrected_pruning_(expert, keep)
    after_fix_hashes = {kind: tensor_sha(getattr(expert, f"{kind}_proj").weight) for kind in ("gate", "up", "down")}
    removed_nonzero_after_fix = {
        "gate": int(torch.count_nonzero(expert.gate_proj.weight[~mask])),
        "up": int(torch.count_nonzero(expert.up_proj.weight[~mask])),
        "down": int(torch.count_nonzero(expert.down_proj.weight[:, ~mask])),
    }
    with torch.no_grad():
        output_after_fix = expert(probe).clone()

    sparse_block_source = inspect.getsource(Qwen3MoeSparseMoeBlock)
    mlp_source = inspect.getsource(Qwen3MoeMLP)
    forward_audit = {
        "experts_container_declared_modulelist": "self.experts = nn.ModuleList" in sparse_block_source,
        "routed_forward_selects_same_expert_modules": "expert_layer = self.experts[expert_idx]" in sparse_block_source and "expert_layer(current_state)" in sparse_block_source,
        "mlp_forward_uses_gate_up_down_modules": all(fragment in mlp_source for fragment in ("self.gate_proj(x)", "self.up_proj(x)", "self.down_proj(")),
        "installed_transformers_source": inspect.getfile(Qwen3MoeSparseMoeBlock),
    }
    mutation_audit = {
        "selection_storage_is_distinct_from_parameter": selection_storage_is_distinct,
        "parameter_hashes_unchanged_after_p9b_helper": before_hashes == after_bug_hashes,
        "removed_nonzero_counts_unchanged_after_p9b_helper": removed_nonzero_before == removed_nonzero_after_bug,
        "expert_forward_bitexact_after_p9b_helper": torch.equal(output_before, output_after_bug),
        "corrected_parameter_hashes_changed": all(after_fix_hashes[kind] != before_hashes[kind] for kind in before_hashes),
        "corrected_removed_regions_are_zero": all(value == 0 for value in removed_nonzero_after_fix.values()),
        "corrected_expert_forward_changes": not torch.equal(output_before, output_after_fix),
        "corrected_forward_different_bf16_elements": int(torch.count_nonzero(output_before.view(torch.uint16) != output_after_fix.view(torch.uint16))),
        "removed_nonzero_before": removed_nonzero_before,
        "removed_nonzero_after_bug": removed_nonzero_after_bug,
        "removed_nonzero_after_fix": removed_nonzero_after_fix,
    }
    checks = {
        "p9b_and_p9e_masks_materially_different": mask_index_mismatches > 1_000_000 and identical_expert_layer_masks == 0,
        "p9b_and_p9e_candidate_outputs_identical": all(output_equalities.values()),
        "p9b_boolean_selections_are_copies": all(selection_storage_is_distinct.values()),
        "p9b_helper_is_noop_on_actual_checkpoint_expert": mutation_audit["parameter_hashes_unchanged_after_p9b_helper"] and mutation_audit["removed_nonzero_counts_unchanged_after_p9b_helper"],
        "p9b_helper_leaves_actual_expert_forward_bitexact": mutation_audit["expert_forward_bitexact_after_p9b_helper"],
        "correct_indexed_fill_mutates_weights_and_forward": mutation_audit["corrected_parameter_hashes_changed"] and mutation_audit["corrected_removed_regions_are_zero"] and mutation_audit["corrected_expert_forward_changes"],
        "qwen_forward_uses_targeted_expert_parameters": all(value for key, value in forward_audit.items() if key != "installed_transformers_source"),
    }
    audit_proven = all(checks.values())
    result = {
        "kind": "gaugepack_p9d1_p9b_mutation_audit",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "p9b_pruning_noop_proven" if audit_proven else "p9b_mutation_audit_inconclusive",
        "inputs": {
            "p9b_script_sha256": sha256(P9B_SCRIPT),
            "p9e_script_sha256": sha256(P9E_SCRIPT),
            "p9b_masks_sha256": sha256(P9B_MASKS),
            "p9e_masks_sha256": sha256(P9E_MASKS),
            "p9b_validation_sha256": sha256(P9B_RESULT),
            "p9e_validation_sha256": sha256(P9E_RESULT),
            "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
        },
        "mask_comparison": {
            "index_position_mismatches": mask_index_mismatches,
            "total_index_positions": 48 * 128 * 384,
            "identical_expert_layer_masks": identical_expert_layer_masks,
            "total_expert_layer_masks": 48 * 128,
        },
        "output_equalities": output_equalities,
        "forward_audit": forward_audit,
        "mutation_audit": mutation_audit,
        "checks": checks,
        "conclusion": "P9B/P9E boolean-index zero_ calls mutate temporary copies, not expert Parameters. Their reported quality is the unpruned Q5 baseline and cannot authorize GaugePack.",
        "required_repair": "Replace advanced-index zero_ with masked_fill_ on each full Parameter, then rerun calibration, validation and test before any GaugePack codec or kernel claim.",
        "runtime": {"seconds": time.perf_counter() - started, "device": "cpu"},
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# GaugePack P9D-1 — audit van de P9B-pruningpremisse\n\n"
        f"Uitkomst: **{result['status']}**.\n\n"
        "De Qwen-forward gebruikt werkelijk de `ModuleList`-experts en hun gate/up/down-Parameters. "
        "Het defect zit in de mutatie: `weight[boolean_mask].zero_()` en "
        "`weight[:, boolean_mask].zero_()` werken op advanced-indexkopieën en schrijven niet terug naar de Parameter.\n\n"
        f"Op echte laag-0/expert-0-checkpointgewichten bleven alle drie SHA-256-hashes en alle aantallen "
        f"niet-nulwaarden na de P9B-helper ongewijzigd; de expert-forward bleef bitexact. Een gecorrigeerde "
        f"`masked_fill_`-mutatie nulde alle bedoelde waarden en veranderde "
        f"{mutation_audit['corrected_forward_different_bf16_elements']:,} BF16-outputelementen in de vaste probe.\n\n"
        f"P9B en P9E verschillen op {mask_index_mismatches:,} van {48 * 128 * 384:,} opgeslagen "
        "indexposities en hebben nul identieke expert-laagmaskers, maar hun candidate CE, top-1, "
        "final-hidden error en alle 48 laag-errors zijn exact gelijk. Dat is nu mechanistisch verklaard.\n\n"
        "Gevolg: de P9B-kwaliteitspass bewijst geen veilige 50%-pruning. P9C blijft een werkelijk uitgevoerde "
        "andere, destructieve compact/requantize-proef, maar kan P9B niet valideren. GaugePack P9D-1 is "
        "geblokkeerd tot P9B-R met een echte in-place mutatie validation én test passeert.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
