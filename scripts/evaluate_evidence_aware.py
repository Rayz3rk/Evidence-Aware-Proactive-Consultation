"""Audit and evaluate a trained evidence-aware checkpoint.

The default mode reports label/data integrity and route/slot metrics on the
same pair-grouped development split used by training.  Generation is optional
because it is slower and should not be confused with route/slot accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proactive_consultation.evidence_aware import (  # noqa: E402
    IGNORE_INDEX,
    ROUTE_LABELS,
    EvidenceAwareCollator,
    EvidenceAwareDataset,
    build_examples,
    build_slot_vocabulary,
    load_slot_labels,
    route_for_row,
)
from proactive_consultation.scope3000 import load_views, split_by_pair_id  # noqa: E402
from proactive_consultation.evidence_aware_runtime import EvidenceAwareRuntime  # noqa: E402


def metric_block(y_true: list[int], y_pred: list[int], labels: list[int]) -> dict[str, Any]:
    confusion = {str(label): {str(other): 0 for other in labels} for label in labels}
    for true, pred in zip(y_true, y_pred):
        confusion[str(true)][str(pred)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = confusion[str(label)][str(label)]
        fp = sum(confusion[str(other)][str(label)] for other in labels if other != label)
        fn = sum(confusion[str(label)][str(other)] for other in labels if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[str(label)] = {"support": tp + fn, "precision": precision, "recall": recall, "f1": f1}
    return {
        "count": len(y_true),
        "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "confusion": confusion,
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, default=ROOT / "outputs" / "mixed_dialogue_full_v4_20260917" / "releases" / "scope3000" / "sft_candidates.auto.jsonl")
    parser.add_argument("--audit-file", type=Path, default=ROOT / "outputs" / "mixed_dialogue_full_v4_20260917" / "releases" / "scope3000" / "paired_contexts.auto.jsonl")
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--quality-status", action="append", default=None, help="Allow this release quality_status; repeat for multiple statuses")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--generate-limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    quality_statuses = set(args.quality_status or {"human_review_passed"})
    rows = load_views(args.data_file, "joint", allowed_quality_statuses=quality_statuses)
    pair_slots = load_slot_labels(args.audit_file)
    inquiry_rows = [row for row in rows if row["task_type"] == "evidence_missing_question"]
    missing_slots = [row["pair_id"] for row in inquiry_rows if row["pair_id"] not in pair_slots]
    if missing_slots:
        raise ValueError(f"Missing slots for {len(missing_slots)} inquiry rows: {missing_slots[:5]}")
    slot_vocab = build_slot_vocabulary(rows, pair_slots)
    slot_to_id = {slot: i for i, slot in enumerate(slot_vocab)}
    train_rows, dev_rows = split_by_pair_id(rows, args.dev_fraction)
    if args.max_samples is not None:
        dev_rows = dev_rows[:args.max_samples]

    route_distribution = Counter(route_for_row(row) for row in rows)
    slot_distribution = Counter(pair_slots[row["pair_id"]] for row in inquiry_rows)
    pair_task_types: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        pair_task_types[row["pair_id"]].add(row["task_type"])
    duplicate_pairs = {pair_id: sorted(tasks) for pair_id, tasks in pair_task_types.items() if len(tasks) > 1}

    runtime = EvidenceAwareRuntime(args.checkpoint)
    runtime.load()
    assert runtime.processor is not None and runtime.model is not None and runtime.tokenizer is not None
    max_length = int(json.loads((args.checkpoint / "evidence_aware_manifest.json").read_text(encoding="utf-8")).get("max_length", 512))
    examples = build_examples(runtime.processor, dev_rows, pair_slots, slot_to_id, max_length=max_length)
    loader = DataLoader(EvidenceAwareDataset(examples), batch_size=args.batch_size, shuffle=False, collate_fn=EvidenceAwareCollator(runtime.tokenizer.pad_token_id))

    route_true: list[int] = []
    route_pred: list[int] = []
    slot_true: list[int] = []
    slot_pred: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            device = runtime._input_device()
            moved = {key: value.to(device) for key, value in batch.items()}
            output = runtime.model(**moved)
            route_true.extend(batch["route_labels"].tolist())
            route_pred.extend(output["route_logits"].argmax(dim=-1).detach().cpu().tolist())
            valid = batch["slot_labels"].ne(IGNORE_INDEX)
            if valid.any():
                slot_true.extend(batch["slot_labels"][valid].tolist())
                slot_pred.extend(output["slot_logits"][valid.to(output["slot_logits"].device)].argmax(dim=-1).detach().cpu().tolist())

    id_to_route = {index: name for name, index in runtime.route_labels.items()}
    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data": {
            "all_rows": len(rows),
            "allowed_quality_statuses": sorted(quality_statuses),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "inquiry_rows": len(inquiry_rows),
            "missing_slot_rows": len(missing_slots),
            "route_distribution": dict(route_distribution),
            "slot_distribution": dict(slot_distribution),
            "slot_vocabulary": slot_vocab,
            "pair_ids_with_both_view_types": len(duplicate_pairs),
            "pair_type_examples": dict(list(duplicate_pairs.items())[:10]),
        },
        "route": metric_block(route_true, route_pred, sorted(id_to_route)),
        "slot": metric_block(slot_true, slot_pred, list(range(len(slot_vocab)))),
        "route_labels": id_to_route,
    }
    if args.generate_limit > 0:
        generated = []
        for row in dev_rows[:args.generate_limit]:
            messages = [{"role": "user" if item["role"] == "user" else "assistant", "content": item["content"]} for item in row["messages"]]
            result = runtime.generate(messages)
            generated.append({"pair_id": row["pair_id"], "gold_route": route_for_row(row), **result.as_dict()})
        report["generation_samples"] = generated
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
