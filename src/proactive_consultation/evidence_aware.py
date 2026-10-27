"""Evidence-aware routing, slot prediction, and conditional generation.

The model has one shared causal-language backbone and lightweight heads that
predict the response mode and, for a proactive inquiry, the information slot.
Construction-only audit fields are used to create labels but are never copied
into model-visible messages.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .token_limits import token_limit

ANSWER_ROUTE = "answer"
ASK_ROUTE = "ask"
ANSWER_AND_ASK_ROUTE = "answer_and_ask"
ROUTE_LABELS = {
    ANSWER_ROUTE: 0,
    ASK_ROUTE: 1,
    ANSWER_AND_ASK_ROUTE: 2,
}
ROUTE_NAMES = {value: key for key, value in ROUTE_LABELS.items()}
ANSWER_CONTROL = "\n[ROUTE=ANSWER]\n"
ASK_CONTROL = "\n[ROUTE=ASK]\n"
ANSWER_AND_ASK_CONTROL = "\n[ROUTE=ANSWER_AND_ASK]\n"
SLOT_PREFIX = "[INQUIRY_SLOT="
SLOT_SUFFIX = "]"
IGNORE_INDEX = -100

# This is intentionally conservative.  A normal response containing an
# explicit question mark is a mixed answer-and-ask example, not a pure answer.
_QUESTION_END_RE = re.compile(
    r"(?:吗|嘛|呢|么|哪(?:里|儿|些|个)|什么|谁|为何|为什么|如何|怎么|多少|几时|何时|是否|有没有|有无|能否|可否|是不是|还是)\s*[。.!！?？]?$"
)


def is_question_like(completion: str) -> bool:
    """Return whether a completion contains an explicit follow-up question.

    The released normal-dialogue data contains some turns that both explain
    and ask.  Treating these as pure answers creates avoidable route-label
    noise, so the route head receives a third ``answer_and_ask`` label.
    """
    text = str(completion or "").strip()
    if not text:
        return False
    if "?" in text or "？" in text:
        return True
    # Check sentence tails rather than every occurrence of ``是否`` etc.; this
    # avoids labelling explanatory phrases such as ``是否需要进一步观察`` as a
    # question unless they form a standalone interrogative sentence.
    for sentence in re.split(r"[。.!！;；\n]", text):
        if _QUESTION_END_RE.search(sentence.strip()):
            return True
    return False


def route_for_row(row: Mapping[str, Any]) -> str:
    """Derive a route label from a retained training view."""
    if row.get("task_type") == "evidence_missing_question":
        return ASK_ROUTE
    if row.get("task_type") == "normal_dialogue":
        return ANSWER_AND_ASK_ROUTE if is_question_like(str(row.get("completion", ""))) else ANSWER_ROUTE
    raise ValueError(f"Unsupported task_type={row.get('task_type')!r}")


def slot_token(slot: str) -> str:
    """Convert an audit slot name into a stable internal control string."""
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in slot.strip())
    if not safe:
        raise ValueError(f"Invalid empty slot: {slot!r}")
    return f"{SLOT_PREFIX}{safe}{SLOT_SUFFIX}"


def load_slot_labels(path: Path) -> dict[str, str]:
    """Load pair-to-slot labels from audit metadata without exposing its text."""
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = row.get("pair_id")
            plan = row.get("construction_plan") or {}
            fact = plan.get("fact") or {}
            slot = fact.get("slot")
            if not isinstance(pair_id, str) or not pair_id:
                raise ValueError(f"Missing pair_id at {path}:{line_no}")
            if slot is None:
                continue
            if not isinstance(slot, str) or not slot.strip():
                raise ValueError(f"Invalid slot at {path}:{line_no}")
            slot = slot.strip()
            previous = labels.get(pair_id)
            if previous is not None and previous != slot:
                raise ValueError(f"Conflicting slot labels for pair_id={pair_id!r}")
            labels[pair_id] = slot
    return labels


def build_slot_vocabulary(rows: Sequence[dict[str, Any]], pair_slots: Mapping[str, str]) -> list[str]:
    """Return a deterministic vocabulary of slots occurring in inquiry views."""
    return sorted({
        pair_slots[row["pair_id"]]
        for row in rows
        if row.get("task_type") == "evidence_missing_question"
        and row.get("pair_id") in pair_slots
    })


def add_control_tokens(_tokenizer: Any, slots: Sequence[str]) -> list[str]:
    """Return internal controls without mutating the base vocabulary."""
    return [
        ANSWER_CONTROL,
        ASK_CONTROL,
        ANSWER_AND_ASK_CONTROL,
        *[slot_token(slot) for slot in slots],
    ]


def _messages_to_text(processor: Any, messages: Sequence[Mapping[str, str]]) -> str:
    return processor.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=True
    )


def encode_context_with_controls(
    processor: Any,
    messages: Sequence[Mapping[str, str]],
    controls: Sequence[str] = (),
) -> tuple[list[int], int, int]:
    """Encode context and internal controls as separate token fragments."""
    tokenizer = processor.tokenizer
    context_text = _messages_to_text(processor, messages)
    context_ids = tokenizer(context_text, add_special_tokens=False)["input_ids"]
    control_text = "".join(controls)
    control_ids = tokenizer(control_text, add_special_tokens=False)["input_ids"] if control_text else []
    return list(context_ids) + list(control_ids), len(context_ids), len(context_ids) + len(control_ids)


def _encode_completion(tokenizer: Any, completion: str) -> list[int]:
    return list(tokenizer(completion, add_special_tokens=False)["input_ids"])


def build_examples(
    processor: Any,
    rows: Sequence[dict[str, Any]],
    pair_slots: Mapping[str, str],
    slot_to_id: Mapping[str, int],
    max_length: int | None = None,
    control_mode: str = "route_slot",
) -> list[dict[str, Any]]:
    """Build tokenized examples while excluding audit content from prompts."""
    max_length = token_limit(max_length)
    examples: list[dict[str, Any]] = []
    tokenizer = processor.tokenizer
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    for row in rows:
        pair_id = row["pair_id"]
        route = route_for_row(row)
        if control_mode not in {"route_slot", "route_only", "slot_only", "none"}:
            raise ValueError(f"Unsupported control_mode={control_mode!r}")
        if route == ASK_ROUTE and pair_id not in pair_slots:
            raise ValueError(f"Missing slot label for inquiry pair {pair_id!r}")
        if control_mode == "route_slot":
            if route == ASK_ROUTE:
                controls = [ASK_CONTROL, slot_token(pair_slots[pair_id])]
            elif route == ANSWER_AND_ASK_ROUTE:
                controls = [ANSWER_AND_ASK_CONTROL]
            else:
                controls = [ANSWER_CONTROL]
        elif control_mode == "route_only":
            controls = {
                ASK_ROUTE: [ASK_CONTROL],
                ANSWER_AND_ASK_ROUTE: [ANSWER_AND_ASK_CONTROL],
                ANSWER_ROUTE: [ANSWER_CONTROL],
            }[route]
        elif control_mode == "slot_only":
            controls = [slot_token(pair_slots[pair_id])] if route == ASK_ROUTE else []
        else:
            controls = []
        prompt_ids, context_length, prompt_length = encode_context_with_controls(
            processor, row["messages"], controls
        )
        completion_ids = _encode_completion(tokenizer, str(row["completion"]))
        if not completion_ids:
            raise ValueError(f"Empty completion for pair {pair_id!r}")

        if max_length is not None and len(completion_ids) + len(prompt_ids) + len(eos_ids) > max_length:
            control_length = prompt_length - context_length
            if max_length <= control_length + 1:
                raise ValueError("max_length is too small for controls and a target")
            max_completion = max(1, max_length - control_length - 1)
            completion_ids = completion_ids[:max_completion]
            budget_for_context = max_length - len(completion_ids) - len(eos_ids) - control_length
            budget_for_context = max(1, budget_for_context)
            context_ids = prompt_ids[:context_length][-budget_for_context:]
            control_ids = prompt_ids[context_length:prompt_length]
            prompt_ids = context_ids + control_ids
            context_length = len(context_ids)
            prompt_length = len(prompt_ids)

        full_ids = prompt_ids + completion_ids + eos_ids
        if len(full_ids) <= prompt_length:
            raise ValueError(f"Completion vanished for pair {pair_id!r}")
        examples.append({
            "input_ids": full_ids,
            "prompt_length": prompt_length,
            "context_length": context_length,
            "route_label": ROUTE_LABELS[route],
            "slot_label": slot_to_id.get(pair_slots.get(pair_id, ""), IGNORE_INDEX) if route == ASK_ROUTE else IGNORE_INDEX,
            "pair_id": pair_id,
            "task_type": row["task_type"],
            "route": route,
        })
    return examples


class EvidenceAwareDataset(Dataset):
    def __init__(self, examples: Sequence[dict[str, Any]]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


@dataclass
class EvidenceAwareCollator:
    pad_token_id: int

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        context_lengths: list[int] = []
        prompt_lengths: list[int] = []
        route_labels: list[int] = []
        slot_labels: list[int] = []
        for item in features:
            ids = list(item["input_ids"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            target = [IGNORE_INDEX] * item["prompt_length"] + ids[item["prompt_length"]:]
            labels.append(target + [IGNORE_INDEX] * pad)
            context_lengths.append(item["context_length"])
            prompt_lengths.append(item["prompt_length"])
            route_labels.append(item["route_label"])
            slot_labels.append(item["slot_label"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "context_lengths": torch.tensor(context_lengths, dtype=torch.long),
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "route_labels": torch.tensor(route_labels, dtype=torch.long),
            "slot_labels": torch.tensor(slot_labels, dtype=torch.long),
        }


class EvidenceAwareModel(nn.Module):
    """Causal LM with an explicit route-to-slot decision chain.

    The shared backbone encodes the context ``c``.  The route head predicts
    ``z`` first; the slot head then receives a route-conditioned representation
    together with the pooled context and predicts ``s``.  Training uses a
    differentiable route posterior, while inference uses the selected route,
    so the standalone inquiry path is ``c -> z -> s -> generation``.  A mixed
    response reuses this ASK-conditioned slot head on ``c + generated answer``
    before producing its follow-up; mixed training views retain masked slots.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_slots: int,
        route_loss_weight: float = 0.5,
        slot_loss_weight: float = 0.5,
        route_labels: Mapping[str, int] | None = None,
        route_class_weights: Sequence[float] | None = None,
        slot_class_weights: Sequence[float] | None = None,
        slot_names: Sequence[str] | None = None,
        slot_condition: str = "route",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(backbone.config, "text_config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot infer backbone hidden size")
        self.hidden_size = int(hidden_size)
        self.route_labels = dict(route_labels or ROUTE_LABELS)
        self.num_routes = max(self.route_labels.values()) + 1
        self.route_head = nn.Linear(self.hidden_size, self.num_routes)
        self.route_embedding = nn.Embedding(self.num_routes, self.hidden_size)
        self.slot_head = nn.Linear(self.hidden_size * (2 if slot_condition == "route" else 1), num_slots) if num_slots else None
        self.route_loss_weight = float(route_loss_weight)
        self.slot_loss_weight = float(slot_loss_weight)
        self.num_slots = int(num_slots)
        self.slot_names = list(slot_names or [])
        self.slot_condition = str(slot_condition)
        if self.slot_condition not in {"route", "context"}:
            raise ValueError("slot_condition must be route or context")
        if self.slot_names and len(self.slot_names) != self.num_slots:
            raise ValueError("slot_names length must match num_slots")
        self.config = backbone.config
        route_weights = list(route_class_weights) if route_class_weights is not None else [1.0] * self.num_routes
        slot_weights = list(slot_class_weights) if slot_class_weights is not None else [1.0] * self.num_slots
        if len(route_weights) != self.num_routes:
            raise ValueError("route_class_weights length must match route labels")
        if len(slot_weights) != self.num_slots:
            raise ValueError("slot_class_weights length must match slot vocabulary")
        self.register_buffer("route_class_weights", torch.tensor(route_weights, dtype=torch.float32), persistent=True)
        self.register_buffer("slot_class_weights", torch.tensor(slot_weights, dtype=torch.float32), persistent=True)

    def _pool_context(self, hidden: torch.Tensor, context_lengths: torch.Tensor) -> torch.Tensor:
        indices = (context_lengths.to(hidden.device) - 1).clamp_min(0)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, indices]

    @staticmethod
    def _head_forward(head: nn.Linear, pooled: torch.Tensor) -> torch.Tensor:
        return F.linear(
            pooled.to(device=head.weight.device, dtype=head.weight.dtype),
            head.weight,
            head.bias,
        )

    def _route_condition_embedding(
        self,
        route_logits: torch.Tensor,
        hard: bool = False,
        route_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the route-conditioned representation for slot prediction."""
        embedding_weight = self.route_embedding.weight.to(
            device=route_logits.device, dtype=route_logits.dtype
        )
        if route_ids is not None:
            if route_ids.shape != route_logits.shape[:-1]:
                raise ValueError("slot_route_ids must contain one route id per context")
            route_ids = route_ids.to(device=route_logits.device, dtype=torch.long)
            if ((route_ids < 0) | (route_ids >= self.num_routes)).any():
                raise ValueError("slot_route_ids contains an unknown route id")
            return F.embedding(route_ids, embedding_weight)
        if hard:
            route_ids = route_logits.argmax(dim=-1)
            return F.embedding(route_ids, embedding_weight)
        probabilities = torch.softmax(route_logits, dim=-1)
        return probabilities @ embedding_weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        context_lengths: torch.Tensor | None = None,
        route_labels: torch.Tensor | None = None,
        slot_labels: torch.Tensor | None = None,
        prompt_lengths: torch.Tensor | None = None,
        hard_route_condition: bool = False,
        slot_route_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del prompt_lengths
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        )
        logits = outputs.logits
        hidden = outputs.hidden_states[-1]
        result: dict[str, Any] = {"logits": logits}
        if context_lengths is not None:
            pooled = self._pool_context(hidden, context_lengths)
            route_logits = self._head_forward(self.route_head, pooled)
            result["route_logits"] = route_logits
            if self.slot_head is not None:
                if self.slot_condition == "route":
                    # A mixed response reuses the ASK-trained slot head after
                    # its answer stage; this override does not relabel the
                    # top-level route or add slot supervision to mixed views.
                    route_embedding = self._route_condition_embedding(
                        route_logits, hard=hard_route_condition, route_ids=slot_route_ids
                    )
                    slot_input = torch.cat(
                        [
                            pooled.to(device=self.slot_head.weight.device, dtype=self.slot_head.weight.dtype),
                            route_embedding.to(device=self.slot_head.weight.device, dtype=self.slot_head.weight.dtype),
                        ],
                        dim=-1,
                    )
                else:
                    slot_input = pooled.to(device=self.slot_head.weight.device, dtype=self.slot_head.weight.dtype)
                result["slot_logits"] = self._head_forward(self.slot_head, slot_input)
                result["slot_condition_route"] = (
                    slot_route_ids.to(route_logits.device).detach()
                    if slot_route_ids is not None else route_logits.argmax(dim=-1).detach()
                )
            else:
                result["slot_logits"] = None

        losses: list[torch.Tensor] = []
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            valid = shift_labels.ne(IGNORE_INDEX)
            lm_loss = F.cross_entropy(shift_logits[valid], shift_labels[valid]) if valid.any() else shift_logits.sum() * 0.0
            result["lm_loss"] = lm_loss
            losses.append(lm_loss)
        if route_labels is not None and context_lengths is not None:
            route_logits = result["route_logits"]
            route_loss = F.cross_entropy(
                route_logits,
                route_labels.to(route_logits.device),
                weight=self.route_class_weights.to(route_logits.device, dtype=route_logits.dtype),
            )
            result["route_loss"] = route_loss
            losses.append(self.route_loss_weight * route_loss)
        if slot_labels is not None and context_lengths is not None and self.slot_head is not None:
            slot_logits = result["slot_logits"]
            slot_labels_device = slot_labels.to(slot_logits.device)
            valid = slot_labels_device.ne(IGNORE_INDEX)
            slot_loss = (
                F.cross_entropy(
                    slot_logits[valid],
                    slot_labels_device[valid],
                    weight=self.slot_class_weights.to(slot_logits.device, dtype=slot_logits.dtype),
                )
                if valid.any()
                else slot_logits.sum() * 0.0
            )
            result["slot_loss"] = slot_loss
            result["slot_supervised_count"] = valid.sum().detach()
            losses.append(self.slot_loss_weight * slot_loss)
        if losses:
            result["loss"] = sum(losses)
        return result

    @torch.no_grad()
    def predict_route_and_slot(self, **batch: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.eval()
        # Do not pass gold route labels here: deployment must condition slot
        # prediction on the route selected by the route head.
        batch = dict(batch)
        batch.pop("route_labels", None)
        batch.pop("slot_labels", None)
        batch["hard_route_condition"] = True
        outputs = self.forward(**batch)
        route = outputs["route_logits"].argmax(dim=-1)
        slot = outputs["slot_logits"].argmax(dim=-1) if outputs.get("slot_logits") is not None else None
        return route, slot


def make_control_tokens(route: int, slot: str | None = None, control_mode: str = "route_slot") -> list[str]:
    if control_mode == "none":
        return []
    if route == ROUTE_LABELS[ANSWER_ROUTE]:
        return [ANSWER_CONTROL] if control_mode in {"route_slot", "route_only"} else []
    if route == ROUTE_LABELS[ASK_ROUTE]:
        if control_mode == "route_only":
            return [ASK_CONTROL]
        if control_mode == "slot_only":
            if slot is None:
                raise ValueError("Ask route requires a slot")
            return [slot_token(slot)]
        if control_mode == "route_slot":
            if slot is None:
                raise ValueError("Ask route requires a slot")
            return [ASK_CONTROL, slot_token(slot)]
        raise ValueError(f"Unknown control_mode={control_mode!r}")
    if route == ROUTE_LABELS[ANSWER_AND_ASK_ROUTE]:
        return [ANSWER_AND_ASK_CONTROL] if control_mode in {"route_slot", "route_only"} else []
    raise ValueError(f"Unknown route id: {route}")
