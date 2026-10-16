"""Route a conversation as answer/ask and generate the selected next turn."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proactive_consultation.evidence_aware_runtime import EvidenceAwareRuntime  # noqa: E402


def load_json_messages(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("messages JSON must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            raise ValueError("each message must contain exactly role and content")
        role = item["role"]
        if role == "patient":
            role = "user"
        elif role == "doctor":
            role = "assistant"
        if role not in {"user", "assistant", "system"} or not isinstance(item["content"], str):
            raise ValueError("invalid message")
        normalized.append({"role": role, "content": item["content"]})
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--messages-json", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None, help="No limit by default; 0 disables the limit")
    parser.add_argument("--max-input-tokens", type=int, default=None, help="No limit by default; 0 disables the limit")
    parser.add_argument("--route-threshold", type=float, default=0.5)
    args = parser.parse_args()
    runtime = EvidenceAwareRuntime(
        args.checkpoint,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        route_threshold=args.route_threshold,
    )
    result = runtime.generate(load_json_messages(args.messages_json))
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
