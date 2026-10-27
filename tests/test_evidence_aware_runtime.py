from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch
from torch import nn

from proactive_consultation.evidence_aware import (
    ANSWER_AND_ASK_ROUTE,
    ANSWER_ROUTE,
    ASK_ROUTE,
    ROUTE_LABELS,
    make_control_tokens,
)
from proactive_consultation.evidence_aware_runtime import EvidenceAwareRuntime


class RecordingTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) % 31 for character in text]}


class RecordingProcessor:
    def __init__(self) -> None:
        self.tokenizer = RecordingTokenizer()
        self.contexts: list[list[dict[str, str]]] = []

    def apply_chat_template(
        self, messages: Sequence[Mapping[str, str]], **kwargs: Any
    ) -> str:
        self.contexts.append([dict(message) for message in messages])
        return "".join(f"<{message['role']}>{message['content']}" for message in messages)


class RecordingModel(nn.Module):
    def __init__(self, slot_head_enabled: bool = True) -> None:
        super().__init__()
        self.slot_head = nn.Linear(1, 2) if slot_head_enabled else None
        self.calls: list[dict[str, Any]] = []

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(kwargs)
        return {"slot_logits": torch.tensor([[0.0, 3.0]])}


class RecordingRuntime(EvidenceAwareRuntime):
    def __init__(
        self,
        responses: Sequence[str],
        *,
        slot_head_enabled: bool = True,
        control_mode: str = "route_slot",
        route: str = ANSWER_AND_ASK_ROUTE,
    ) -> None:
        super().__init__("unused-test-checkpoint")
        self._loaded = True
        self.processor = RecordingProcessor()
        self.tokenizer = self.processor.tokenizer
        self.model = RecordingModel(slot_head_enabled)
        self.slots = ["duration", "aggravating_factor"]
        self.control_mode = control_mode
        self.responses = iter(responses)
        self.top_level_route = route
        self.generation_calls: list[tuple[list[dict[str, str]], list[str]]] = []

    def _input_device(self) -> torch.device:
        return torch.device("cpu")

    def predict_route_and_slot(self, messages: Sequence[Mapping[str, str]]) -> tuple:
        return self.top_level_route, None, 0.1, 0.8, None, {self.top_level_route: 0.8}

    def _generate_completion(
        self, messages: Sequence[Mapping[str, str]], controls: Sequence[str]
    ) -> str:
        self.generation_calls.append(([dict(message) for message in messages], list(controls)))
        return next(self.responses)


def test_mixed_follow_up_predicts_slot_from_context_and_generated_answer() -> None:
    context = [{"role": "user", "content": "I have had a cough for three days."}]
    runtime = RecordingRuntime(["Please rest.", "When is your cough worse?"])
    result = runtime.generate(context)

    assert result.route == ANSWER_AND_ASK_ROUTE
    assert result.slot == "aggravating_factor"
    assert result.slot_confidence == pytest.approx(0.952574, abs=1e-6)
    assert result.completion == "Please rest. When is your cough worse?"
    assert result.as_dict()["slot"] == "aggravating_factor"
    assert runtime.processor.contexts == [context + [{"role": "assistant", "content": "Please rest."}]]
    assert len(runtime.model.calls) == 1
    assert runtime.model.calls[0]["slot_route_ids"].tolist() == [ROUTE_LABELS[ASK_ROUTE]]
    assert runtime.generation_calls[0][1] == make_control_tokens(ROUTE_LABELS[ANSWER_ROUTE])
    assert runtime.generation_calls[1][1] == make_control_tokens(
        ROUTE_LABELS[ASK_ROUTE], "aggravating_factor"
    )
    assert runtime.generation_calls[1][0][-2] == {"role": "assistant", "content": "Please rest."}
    assert runtime.generation_calls[1][0][-1]["role"] == "user"
    assert context == [{"role": "user", "content": "I have had a cough for three days."}]


def test_empty_answer_still_selects_a_slot_before_asking() -> None:
    context = [{"role": "user", "content": "Cough."}]
    runtime = RecordingRuntime(["", "When is it worse?"])
    result = runtime.generate(context)

    assert result.completion == "When is it worse?"
    assert result.slot == "aggravating_factor"
    assert runtime.processor.contexts == [context]
    assert all(message.get("content") for message in runtime.generation_calls[1][0])
    assert runtime.generation_calls[1][1] == make_control_tokens(
        ROUTE_LABELS[ASK_ROUTE], "aggravating_factor"
    )


@pytest.mark.parametrize("control_mode", ["route_slot", "route_only", "slot_only", "none"])
def test_disabled_slot_head_uses_explicit_no_slot_ablation_fallback(control_mode: str) -> None:
    runtime = RecordingRuntime(
        ["Please rest.", "Any fever?"], slot_head_enabled=False, control_mode=control_mode
    )
    result = runtime.generate([{"role": "user", "content": "Cough."}])

    assert result.route == ANSWER_AND_ASK_ROUTE
    assert result.slot is None and result.slot_confidence is None
    assert runtime.model.calls == []
    assert runtime.generation_calls[1][1] == make_control_tokens(
        ROUTE_LABELS[ANSWER_AND_ASK_ROUTE], None, control_mode
    )


def test_no_control_ablation_does_not_inject_predicted_slot() -> None:
    runtime = RecordingRuntime(["Please rest.", "Any fever?"], control_mode="none")
    result = runtime.generate([{"role": "user", "content": "Cough."}])

    assert result.slot == "aggravating_factor"
    assert len(runtime.model.calls) == 1
    assert all(controls == [] for _, controls in runtime.generation_calls)


def test_standalone_ask_without_a_slot_keeps_existing_error() -> None:
    runtime = RecordingRuntime([], slot_head_enabled=False, route=ASK_ROUTE)
    with pytest.raises(RuntimeError, match="Ask route was selected but no slot head"):
        runtime.generate([{"role": "user", "content": "Cough."}])


def test_answer_route_does_not_run_follow_up_slot_prediction() -> None:
    runtime = RecordingRuntime(["Please rest."], route=ANSWER_ROUTE)
    result = runtime.generate([{"role": "user", "content": "Cough."}])

    assert result.completion == "Please rest."
    assert result.slot is None and result.slot_confidence is None
    assert runtime.model.calls == []
    assert len(runtime.generation_calls) == 1
