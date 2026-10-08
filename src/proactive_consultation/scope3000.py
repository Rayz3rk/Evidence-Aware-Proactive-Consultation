"""Scope release training data adapter.

The public training view is deliberately treated as a natural next-turn
conversation.  ``task_type`` is retained for offline filtering and reporting,
never copied into a model-visible message.  This lets one joint adapter learn
whether the next turn should be a normal doctor response or a concise question
from the dialogue context itself.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
SCOPE3000_DIR = ROOT / "outputs" / "mixed_dialogue_full_v4_20260917" / "releases" / "scope3000"
SFT_PATH = SCOPE3000_DIR / "sft_candidates.auto.jsonl"
ALLOWED_FIELDS = {
    "source",
    "pair_id",
    "split",
    "quality_status",
    "target_origin",
    "task_type",
    "messages",
    "completion",
}
TASK_TYPES = {"normal_dialogue", "evidence_missing_question"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def _validate_messages(messages: Any, *, row_index: int) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Row {row_index} has no messages")
    clean: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"Row {row_index} contains a non-whitelisted message")
        role, content = message["role"], message["content"]
        if role not in {"user", "assistant", "system"} or not isinstance(content, str):
            raise ValueError(f"Row {row_index} contains an invalid message")
        clean.append({"role": role, "content": content})
    # Source conversations can contain consecutive assistant turns. Preserve
    # them exactly; the adapter must not rewrite the observed dialogue.
    if clean[-1]["role"] == "system":
        raise ValueError(f"Row {row_index} cannot end with a system turn")
    return clean


def load_views(
    path: Path = SFT_PATH,
    task_mode: str = "joint",
    allowed_quality_statuses: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load a release's allowlisted SFT view without exposing audit fields.

    The default remains the human-reviewed scope3000 release. Larger research
    releases can explicitly opt into their recorded quality status, e.g.
    ``{"auto_pass_pending_human"}``; this is never done implicitly.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if task_mode not in {"joint", "normal_only", "question_only"}:
        raise ValueError(f"Unsupported task mode: {task_mode}")
    rows = _read_jsonl(path)
    quality_statuses = allowed_quality_statuses or {"human_review_passed"}
    selected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if set(row) != ALLOWED_FIELDS:
            raise ValueError(f"Row {index} has fields outside the SFT allowlist: {sorted(set(row) - ALLOWED_FIELDS)}")
        if row["task_type"] not in TASK_TYPES:
            raise ValueError(f"Row {index} has unsupported task_type={row['task_type']!r}")
        if row.get("quality_status") not in quality_statuses:
            continue
        if not row.get("pair_id") or not isinstance(row.get("completion"), str) or not row["completion"].strip():
            raise ValueError(f"Row {index} has missing pair_id or completion")
        messages = _validate_messages(row["messages"], row_index=index)
        if task_mode == "normal_only" and row["task_type"] != "normal_dialogue":
            continue
        if task_mode == "question_only" and row["task_type"] != "evidence_missing_question":
            continue
        selected.append({
            "source": row["source"],
            "pair_id": row["pair_id"],
            "task_type": row["task_type"],
            "messages": messages,
            "completion": row["completion"].strip(),
        })
    if not selected:
        raise ValueError(f"No rows selected from {path} for task_mode={task_mode}")
    return selected


def split_by_pair_id(rows: list[dict[str, Any]], dev_fraction: float = 0.1) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split whole pair groups; normal/question siblings never cross splits."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be in (0, 1)")
    pair_ids = sorted({row["pair_id"] for row in rows})
    dev_ids = {
        pair_id for pair_id in pair_ids
        if int(hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF < dev_fraction
    }
    if not dev_ids and len(pair_ids) > 1:
        dev_ids = {pair_ids[-1]}
    train = [row for row in rows if row["pair_id"] not in dev_ids]
    dev = [row for row in rows if row["pair_id"] in dev_ids]
    if not train or not dev:
        raise ValueError("Pair-group split produced an empty train or dev split")
    return train, dev


def _cycle_resample(rows: list[dict[str, Any]], target_size: int, seed: int) -> list[dict[str, Any]]:
    if not rows or target_size <= 0:
        return []
    rng = random.Random(seed)
    result: list[dict[str, Any]] = []
    while len(result) < target_size:
        block = list(rows)
        rng.shuffle(block)
        result.extend(block)
    return result[:target_size]


def mix_training_rows(
    rows: list[dict[str, Any]],
    ratio: str = "2:1",
    seed: int = 42,
    target_total: int | None = None,
) -> list[dict[str, Any]]:
    """Make a deterministic natural or normal:question mixture for joint SFT."""
    if ratio == "natural":
        return list(rows)
    try:
        normal_weight, question_weight = (int(x) for x in ratio.split(":", 1))
    except Exception as exc:
        raise ValueError("ratio must be natural or a positive N:M value") from exc
    if normal_weight <= 0 or question_weight <= 0:
        raise ValueError("ratio weights must be positive")
    normal = [row for row in rows if row["task_type"] == "normal_dialogue"]
    question = [row for row in rows if row["task_type"] == "evidence_missing_question"]
    if not normal or not question:
        raise ValueError("Joint training requires both task types")
    if target_total is not None:
        if target_total <= 0 or target_total % (normal_weight + question_weight) != 0:
            raise ValueError("target_total must be positive and divisible by the ratio weight sum")
        unit = target_total // (normal_weight + question_weight)
        normal_count = unit * normal_weight
        question_count = unit * question_weight
        normal_part = _cycle_resample(normal, normal_count, seed)
        question_part = _cycle_resample(question, question_count, seed + 1)
    else:
        normal_count = len(normal)
        question_count = math.ceil(normal_count * question_weight / normal_weight)
        normal_part = list(normal)
        question_part = _cycle_resample(question, question_count, seed + 1)
    result = normal_part + question_part
    random.Random(seed).shuffle(result)
    return result


def task_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(row["task_type"] for row in rows))


def build_chat_text(processor: Any, row: dict[str, Any]) -> str:
    """Render the raw conversation only; no task instruction or task token is added."""
    messages = list(row["messages"]) + [{"role": "assistant", "content": row["completion"]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
