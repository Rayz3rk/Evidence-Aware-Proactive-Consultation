"""Validate the scope3000 evidence-aware training views before training."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proactive_consultation.evidence_aware import (  # noqa: E402
    build_slot_vocabulary,
    is_question_like,
    load_slot_labels,
    route_for_row,
)
from proactive_consultation.scope3000 import load_views  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=ROOT / "outputs" / "mixed_dialogue_full_v4_20260917" / "releases" / "scope3000" / "sft_candidates.auto.jsonl")
    parser.add_argument("--audit-file", type=Path, default=ROOT / "outputs" / "mixed_dialogue_full_v4_20260917" / "releases" / "scope3000" / "paired_contexts.auto.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quality-status", action="append", default=None, help="Allow this release quality_status; repeat for multiple statuses")
    args = parser.parse_args()

    quality_statuses = set(args.quality_status or {"human_review_passed"})
    rows = load_views(args.data_file, "joint", allowed_quality_statuses=quality_statuses)
    pair_slots = load_slot_labels(args.audit_file)
    inquiries = [row for row in rows if row["task_type"] == "evidence_missing_question"]
    missing_slots = [row["pair_id"] for row in inquiries if row["pair_id"] not in pair_slots]
    route_counts = Counter(route_for_row(row) for row in rows)
    question_like_normal = [row for row in rows if row["task_type"] == "normal_dialogue" and is_question_like(row["completion"])]
    pair_types: defaultdict[str, set[str]] = defaultdict(set)
    pair_sources: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        pair_types[row["pair_id"]].add(row["task_type"])
        pair_sources[row["pair_id"]].add(row["source"])
    conflicts = {pair_id: sorted(types) for pair_id, types in pair_types.items() if len(types) > 2}
    multi_source = {pair_id: sorted(sources) for pair_id, sources in pair_sources.items() if len(sources) > 1}
    report: dict[str, Any] = {
        "status": "pass" if not missing_slots and not conflicts else "fail",
        "allowed_quality_statuses": sorted(quality_statuses),
        "rows": len(rows),
        "unique_pair_ids": len(pair_types),
        "task_counts": dict(Counter(row["task_type"] for row in rows)),
        "route_counts": dict(route_counts),
        "normal_question_like_count": len(question_like_normal),
        "normal_question_like_fraction": len(question_like_normal) / max(1, sum(row["task_type"] == "normal_dialogue" for row in rows)),
        "inquiry_missing_slot_count": len(missing_slots),
        "slot_counts": dict(Counter(pair_slots.get(row["pair_id"], "<missing>") for row in inquiries)),
        "slot_vocabulary": build_slot_vocabulary(rows, pair_slots),
        "pairs_with_both_views": sum(types == {"normal_dialogue", "evidence_missing_question"} for types in pair_types.values()),
        "pairs_with_unexpected_task_types": conflicts,
        "pairs_with_multiple_sources": multi_source,
        "examples": {
            "missing_slot_pair_ids": missing_slots[:20],
            "normal_question_like_pair_ids": [row["pair_id"] for row in question_like_normal[:20]],
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
