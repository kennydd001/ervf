from __future__ import annotations

import gc
import time

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from moe_lab.behavioral import project, rmsnorm, sample_fisher_score_gradients
from moe_lab.moe_layer import load_token_embeddings, loaded_moe_from_official_module
from moe_lab.partial_forward import checkpoint_state_for_prefix, load_decoder_layer
from moe_lab.quantization import fake_quantize_symmetric_per_row_
from moe_lab.reporting import ROOT, envelope, write_json


BLOCK_SIZE = 128
SPLIT_BLOCKS = {"train": 16, "validation": 8, "test": 8}
RANKS = (8, 16, 32, 64, 128, 256)
SEED = 20260809


def corpus_blocks(model_dir, split: str, blocks: int) -> torch.Tensor:
    parquet = (
        ROOT
        / "data"
        / "corpora"
        / "wikitext"
        / "wikitext-2-raw-v1"
        / f"{split}-00000-of-00001.parquet"
    )
    texts = pq.read_table(parquet, columns=["text"])["text"].to_pylist()
    joined = "\n\n".join(text for text in texts if text and text.strip())
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    ids = tokenizer.encode(joined).ids[: blocks * BLOCK_SIZE]
    return torch.tensor(ids, dtype=torch.long).view(blocks, BLOCK_SIZE)


@torch.inference_mode()
def forward_layer(layer, hidden_states):
    batch, sequence, _ = hidden_states.shape
    positions = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
    mask = _prepare_4d_causal_attention_mask(
        None, (batch, sequence), hidden_states, 0
    )
    return layer(
        hidden_states,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        output_attentions=False,
    )[0]


def quantize_routed_three_bit(layer) -> None:
    moe = loaded_moe_from_official_module(layer.mlp, layer=26)
    for expert in moe.experts:
        for weight in (expert.gate, expert.up, expert.down):
            fake_quantize_symmetric_per_row_(weight, 3)


@torch.inference_mode()
def behavioral_metrics(
    teacher_hidden: torch.Tensor,
    candidate_hidden: torch.Tensor,
    token_ids: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
    block_batch: int = 2,
) -> dict[str, float]:
    kl_sum = 0.0
    tokens = 0
    agreement = 0
    teacher_ce_sum = 0.0
    candidate_ce_sum = 0.0
    prediction_tokens = 0
    for start in range(0, teacher_hidden.shape[0], block_batch):
        teacher = teacher_hidden[start : start + block_batch].to(lm_head.device)
        candidate = candidate_hidden[start : start + block_batch].to(lm_head.device)
        ids = token_ids[start : start + block_batch].to(lm_head.device)
        teacher_logits = F.linear(rmsnorm(teacher, norm_weight), lm_head).float()
        candidate_logits = F.linear(rmsnorm(candidate, norm_weight), lm_head).float()
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        candidate_log_probs = F.log_softmax(candidate_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        kl = (teacher_probs * (teacher_log_probs - candidate_log_probs)).sum(dim=-1)
        kl_sum += float(kl.sum().item())
        tokens += kl.numel()
        agreement += int(
            (teacher_logits.argmax(dim=-1) == candidate_logits.argmax(dim=-1))
            .sum()
            .item()
        )
        labels = ids[:, 1:]
        teacher_loss = F.cross_entropy(
            teacher_logits[:, :-1].reshape(-1, teacher_logits.shape[-1]),
            labels.reshape(-1),
            reduction="sum",
        )
        candidate_loss = F.cross_entropy(
            candidate_logits[:, :-1].reshape(-1, candidate_logits.shape[-1]),
            labels.reshape(-1),
            reduction="sum",
        )
        teacher_ce_sum += float(teacher_loss.item())
        candidate_ce_sum += float(candidate_loss.item())
        prediction_tokens += labels.numel()
    teacher_ce = teacher_ce_sum / prediction_tokens
    candidate_ce = candidate_ce_sum / prediction_tokens
    return {
        "teacher_to_candidate_kl": kl_sum / tokens,
        "top1_agreement": agreement / tokens,
        "teacher_cross_entropy": teacher_ce,
        "candidate_cross_entropy": candidate_ce,
        "cross_entropy_delta": candidate_ce - teacher_ce,
        "relative_cross_entropy_delta": (candidate_ce - teacher_ce) / teacher_ce,
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("observability experiment requires CUDA")
    device = torch.device("cuda")
    started = time.perf_counter()
    model_dir = ROOT / "models" / "deepseek-v2-lite"
    split_ids = {
        split: corpus_blocks(model_dir, split, blocks)
        for split, blocks in SPLIT_BLOCKS.items()
    }
    combined_ids = torch.cat(tuple(split_ids.values()), dim=0)
    hidden = load_token_embeddings(model_dir, combined_ids, device)

    for layer_idx in range(27):
        layer, _ = load_decoder_layer(model_dir, layer_idx, device)
        if layer_idx < 26:
            hidden = forward_layer(layer, hidden)
        else:
            teacher_final = forward_layer(layer, hidden)
            quantize_routed_three_bit(layer)
            quantized_final = forward_layer(layer, hidden)
        print(f"layer={layer_idx:02d}", flush=True)
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    split_teacher = {}
    split_quantized = {}
    offset = 0
    for split, blocks in SPLIT_BLOCKS.items():
        split_teacher[split] = teacher_final[offset : offset + blocks].cpu()
        split_quantized[split] = quantized_final[offset : offset + blocks].cpu()
        offset += blocks
    state_path = ROOT / "data" / "traces" / "layer26_teacher_quant3_final_states.safetensors"
    save_file(
        {
            **{f"teacher_{key}": value.contiguous() for key, value in split_teacher.items()},
            **{f"quantized_{key}": value.contiguous() for key, value in split_quantized.items()},
        },
        state_path,
        metadata={
            "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
            "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "layer": "26",
            "quantization": "symmetric per-output-row 3-bit routed experts only",
        },
    )
    del teacher_final, quantized_final, hidden
    gc.collect()
    torch.cuda.empty_cache()

    norm_weight = checkpoint_state_for_prefix(model_dir, "model.norm")["weight"].to(device)
    lm_head = checkpoint_state_for_prefix(model_dir, "lm_head")["weight"].to(device)
    train_hidden = split_teacher["train"].reshape(-1, 2048).to(device)
    gradients = sample_fisher_score_gradients(
        train_hidden, norm_weight, lm_head, batch_size=32, seed=SEED
    ).to(device)
    gram = gradients.T @ gradients / gradients.shape[0]
    trace_g = float(torch.trace(gram).item())
    frobenius_squared = float(gram.square().sum().item())
    effective_rank = (trace_g * trace_g) / max(frobenius_squared, 1e-30)
    torch.manual_seed(SEED)
    _, singular_values, observable_basis = torch.pca_lowrank(
        gradients, q=max(RANKS), center=False, niter=6
    )
    eigenvalues = singular_values.square() / gradients.shape[0]
    captured = {
        str(rank): float(eigenvalues[:rank].sum().item() / trace_g) for rank in RANKS
    }

    train_delta = (
        split_teacher["train"].float() - split_quantized["train"].float()
    ).reshape(-1, 2048).to(device)
    _, _, pca_basis = torch.pca_lowrank(
        train_delta, q=max(RANKS), center=False, niter=6
    )
    random_matrix = torch.randn(2048, max(RANKS), device=device)
    random_basis, _ = torch.linalg.qr(random_matrix, mode="reduced")

    results = {}
    for split in ("validation", "test"):
        teacher = split_teacher[split]
        quantized = split_quantized[split]
        delta = (teacher.float() - quantized.float()).reshape(-1, 2048).to(device)
        baseline = behavioral_metrics(
            teacher, quantized, split_ids[split], norm_weight, lm_head
        )
        methods = {}
        for method, basis in (
            ("behavioral", observable_basis),
            ("error_pca", pca_basis),
            ("random", random_basis),
        ):
            rows = []
            for rank in RANKS:
                correction = project(delta, basis, rank).view_as(teacher).cpu()
                corrected = (quantized.float() + correction.float()).to(torch.bfloat16)
                metrics = behavioral_metrics(
                    teacher, corrected, split_ids[split], norm_weight, lm_head
                )
                baseline_kl = baseline["teacher_to_candidate_kl"]
                metrics["kl_damage_recovery"] = (
                    1.0 - metrics["teacher_to_candidate_kl"] / baseline_kl
                    if baseline_kl > 0
                    else 0.0
                )
                baseline_ce_damage = baseline["cross_entropy_delta"]
                metrics["ce_damage_recovery"] = (
                    1.0 - metrics["cross_entropy_delta"] / baseline_ce_damage
                    if abs(baseline_ce_damage) > 1e-12
                    else 0.0
                )
                rows.append({"rank": rank, **metrics})
            methods[method] = rows
        results[split] = {"quant3_baseline": baseline, "methods": methods}

    report = {
        "status": "complete",
        "experiment": "layer26_h0_future_logit_fisher_observability",
        "model_revision": "604d5664dddd88a0433dbae533b7fe9472482de0",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "split_blocks": SPLIT_BLOCKS,
        "block_size": BLOCK_SIZE,
        "fisher_samples_per_train_token": 1,
        "spectrum": {
            "trace": trace_g,
            "effective_rank": effective_rank,
            "top_eigenvalues": eigenvalues.cpu().tolist(),
            "captured_trace_fraction": captured,
            "r90": next(
                (rank for rank in RANKS if captured[str(rank)] >= 0.90),
                f">{max(RANKS)}",
            ),
        },
        "results": results,
        "state_artifact": str(state_path.resolve()),
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json(
        "layer26_behavioral_observability.json", envelope("behavioral_observability", report)
    )
    print(path)
    print("spectrum", report["spectrum"] | {"top_eigenvalues": eigenvalues[:8].cpu().tolist()})
    for split in ("validation", "test"):
        print(split, "baseline", results[split]["quant3_baseline"])
        for method in ("behavioral", "error_pca", "random"):
            print(split, method)
            for row in results[split]["methods"][method]:
                print(
                    row["rank"],
                    f"KL={row['teacher_to_candidate_kl']:.6f}",
                    f"KLrec={row['kl_damage_recovery']:.3f}",
                    f"CErec={row['ce_damage_recovery']:.3f}",
                )
