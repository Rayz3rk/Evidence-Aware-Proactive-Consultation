"""Generate the fixed 3,000-context main evaluation for serial checkpoints.

This evaluator loads the shared evidence-aware manifest/processor from the run root,
then replaces the model state with a Trainer checkpoint. It preserves the paper's
natural-context-only, greedy, 2,048-input/64-output protocol while recording the
predicted route and slot as auxiliary diagnostics.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proactive_consultation.evidence_aware import (  # noqa: E402
    ASK_ROUTE,
    encode_context_with_controls,
    make_control_tokens,
)
from proactive_consultation.evidence_aware_runtime import EvidenceAwareRuntime  # noqa: E402
from proactive_consultation.scope3000_evaluation import digest, read_rows, messages  # noqa: E402

TEST = ROOT / "example/data/fused_medical/general/test.jsonl"
RUN_ROOT = ROOT / "outputs/checkpoints/evidence_aware_scope2000_repaired_1to1_serial_20epoch_saved"


def clean(text: str) -> str:
    return EvidenceAwareRuntime._clean_completion(text)


def pad_batch(sequences: list[list[int]], pad_id: int, *, left: bool = False) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    max_len = max(len(x) for x in sequences)
    data: list[list[int]] = []
    masks: list[list[int]] = []
    lengths: list[int] = []
    for seq in sequences:
        pad = [pad_id] * (max_len - len(seq))
        if left:
            data.append(pad + seq)
            masks.append([0] * len(pad) + [1] * len(seq))
        else:
            data.append(seq + pad)
            masks.append([1] * len(seq) + [0] * len(pad))
        lengths.append(len(seq))
    return torch.tensor(data, dtype=torch.long), torch.tensor(masks, dtype=torch.long), lengths


def load_checkpoint(checkpoint: Path) -> EvidenceAwareRuntime:
    runtime = EvidenceAwareRuntime(RUN_ROOT, device_map="auto", max_input_tokens=2048, max_new_tokens=64)
    runtime.load()
    assert runtime.model is not None
    state = torch.load(checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True)
    runtime.model.load_state_dict(state, strict=True)
    del state
    runtime.model.eval()
    return runtime


def generate(args: argparse.Namespace) -> None:
    rows = read_rows(TEST)
    if len(rows) != 3000 or len({r["example_id"] for r in rows}) != 3000:
        raise RuntimeError("Expected 3,000 unique fixed test examples")
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "label": args.label,
        "checkpoint": str(args.checkpoint.resolve()),
        "test_sha256": digest(TEST),
        "protocol": {
            "input": "natural_context_only",
            "max_input_tokens": 2048,
            "max_new_tokens": 64,
            "do_sample": False,
            "batch_size": args.batch_size,
            "truncation_side": "left",
            "postprocessing": "strip_reasoning_and_internal_controls",
        },
    }
    manifest_path = output.with_suffix(".generation_manifest.json")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise RuntimeError(f"Generation manifest mismatch: {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    done = read_rows(output)
    if len(done) > len(rows):
        raise RuntimeError("Prediction file has too many rows")
    for old, source in zip(done, rows):
        if old.get("example_id") != source.get("example_id") or old.get("context") != source.get("context"):
            raise RuntimeError("Prediction prefix mismatch; refusing to append")
    if len(done) == len(rows):
        print(json.dumps({"label": args.label, "status": "already_complete", "count": len(done)}, ensure_ascii=False))
        return

    runtime = load_checkpoint(args.checkpoint)
    assert runtime.processor is not None and runtime.tokenizer is not None and runtime.model is not None
    tokenizer = runtime.tokenizer
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("Tokenizer has no pad/eos token")
    device = runtime._input_device()
    names_by_id = {idx: name for name, idx in runtime.route_labels.items()}

    with output.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(len(done), len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            # First pass: route and slot prediction from the unmodified natural context.
            route_sequences: list[list[int]] = []
            route_context_lengths: list[int] = []
            for row in batch:
                ids, context_len, _ = encode_context_with_controls(runtime.processor, messages(row))
                ids, context_len = runtime._trim_context(ids, context_len, context_len)
                route_sequences.append(ids)
                route_context_lengths.append(context_len)
            route_ids, route_mask, _ = pad_batch(route_sequences, int(pad_id), left=False)
            context_lengths = torch.tensor(route_context_lengths, dtype=torch.long, device=device)
            route_ids = route_ids.to(device)
            route_mask = route_mask.to(device)
            with torch.inference_mode():
                route_out = runtime.model(
                    input_ids=route_ids,
                    attention_mask=route_mask,
                    context_lengths=context_lengths,
                    hard_route_condition=True,
                )
            route_probs = torch.softmax(route_out["route_logits"], dim=-1).float().cpu()
            slot_probs = (torch.softmax(route_out["slot_logits"], dim=-1).float().cpu()
                          if route_out.get("slot_logits") is not None else None)
            predictions: list[dict[str, Any]] = []
            generation_sequences: list[list[int]] = []
            route_info: list[tuple[str, str | None, float, float]] = []
            for i, row in enumerate(batch):
                route, ask_prob, route_conf = runtime._route_from_probs(route_probs[i])
                slot = None
                slot_conf = None
                if route == ASK_ROUTE and slot_probs is not None:
                    sid = int(slot_probs[i].argmax())
                    slot = runtime.slots[sid]
                    slot_conf = float(slot_probs[i, sid])
                controls = make_control_tokens(runtime.route_labels[route], slot if route == ASK_ROUTE else None)
                ids, context_len, prompt_len = encode_context_with_controls(runtime.processor, messages(row), controls)
                ids, _ = runtime._trim_context(ids, context_len, prompt_len)
                generation_sequences.append(ids)
                route_info.append((route, slot, ask_prob, route_conf))

            # Second pass: condition generation on the predicted internal controls.
            gen_ids, gen_mask, _ = pad_batch(generation_sequences, int(pad_id), left=True)
            gen_ids = gen_ids.to(device)
            gen_mask = gen_mask.to(device)
            with torch.inference_mode():
                generated = runtime.model.backbone.generate(
                    input_ids=gen_ids,
                    attention_mask=gen_mask,
                    max_new_tokens=64,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for i, row in enumerate(batch):
                route, slot, ask_prob, route_conf = route_info[i]
                continuation = generated[i, gen_ids.shape[1] :]
                prediction = clean(tokenizer.decode(continuation, skip_special_tokens=True))
                route_probs_dict = {name: float(route_probs[i, idx]) for name, idx in runtime.route_labels.items()}
                record = {k: row.get(k) for k in ("example_id", "conversation_id", "source_dataset", "context", "target_question", "target_actions", "label_note")}
                record.update({
                    "prediction": prediction,
                    "prediction_tokens": len(tokenizer.encode(prediction, add_special_tokens=False)),
                    "predicted_route": route,
                    "predicted_slot": slot,
                    "ask_probability": ask_prob,
                    "route_confidence": route_conf,
                    "route_probabilities": route_probs_dict,
                })
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            print(f"{args.label}: generated {start + len(batch)}/{len(rows)}", flush=True)

    del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()

