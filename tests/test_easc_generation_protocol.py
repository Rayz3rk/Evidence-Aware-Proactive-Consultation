"""CPU checks for evaluator routing, resume boundaries and mixed-route costs."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from proactive_consultation.evidence_aware import (
    ANSWER_AND_ASK_ROUTE, ANSWER_ROUTE, ASK_ROUTE, ROUTE_LABELS,
)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) + 100 for char in text]}

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join({10: "answer", 11: "question"}.get(int(token), "") for token in tokens)


class Processor:
    tokenizer = Tokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(f"<{message['role']}>{message['content']}" for message in messages) + "<assistant>"


def evaluators(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import evaluate_serial_main
    import run_mediq_comparison
    return evaluate_serial_main, run_mediq_comparison


@pytest.mark.parametrize("all_mixed", [False, True])
def test_main_batch_preserves_order_and_uses_fresh_mixed_slot(monkeypatch, all_mixed):
    main, _ = evaluators(monkeypatch)
    routes = [ANSWER_AND_ASK_ROUTE] * 4 if all_mixed else [ANSWER_AND_ASK_ROUTE, ANSWER_ROUTE, ASK_ROUTE, ANSWER_AND_ASK_ROUTE]

    class Backbone:
        calls = 0

        def generate(self, input_ids, attention_mask, **kwargs):
            self.calls += 1
            assert input_ids.shape[0] == 2
            return torch.cat([input_ids, torch.tensor([[10, 99], [11, 99]])], dim=1)

    class Model:
        backbone = Backbone()

        def __call__(self, **kwargs):
            logits = torch.full((len(routes), 3), -10.0)
            for index, route in enumerate(routes):
                logits[index, ROUTE_LABELS[route]] = 10.0
            return {"route_logits": logits, "slot_logits": torch.tensor([[10.0, -10.0]] * len(routes))}

    serial_contexts = []

    def serial(messages):
        serial_contexts.append(messages)
        return "serial-" + messages[0]["content"], "onset", 0.8

    runtime = SimpleNamespace(
        processor=Processor(), tokenizer=Tokenizer(), model=Model(), slots=["duration", "onset"],
        route_labels=ROUTE_LABELS, control_mode="route_slot", _input_device=lambda: torch.device("cpu"),
        _trim_context=lambda ids, context_len, prompt_len: (ids, context_len),
        _route_from_probs=lambda p: (routes_by_id[int(p.argmax())], float(p[1]), float(p.max())),
        _generate_answer_and_ask=serial,
    )
    routes_by_id = {index: route for route, index in ROUTE_LABELS.items()}
    rows = [{"context": [{"role": "patient", "text": str(index)}]} for index in range(4)]
    result = main.generate_batch(runtime, rows, 0)
    assert [row["predicted_route"] for row in result] == routes
    for index, route in enumerate(routes):
        if route == ANSWER_AND_ASK_ROUTE:
            assert result[index]["prediction"] == "serial-" + str(index)
            assert result[index]["predicted_slot"] == "onset"
            assert result[index]["slot_confidence"] == 0.8
        elif route == ASK_ROUTE:
            assert result[index]["prediction"] == "question"
            assert result[index]["predicted_slot"] == "duration"
        else:
            assert result[index]["prediction"] == "answer"
            assert result[index]["predicted_slot"] is None
    assert len(serial_contexts) == routes.count(ANSWER_AND_ASK_ROUTE)
    assert runtime.model.backbone.calls == (0 if all_mixed else 1)


@pytest.mark.parametrize("slot_enabled,control_mode", [(True, "route_slot"), (True, "none"), (False, "route_only")])
def test_measured_mixed_route_counts_slot_forward_and_inputs(monkeypatch, tmp_path, slot_enabled, control_mode):
    _, mediq = evaluators(monkeypatch)
    model_calls, prompts = [], []

    class Backbone:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def generate(self, input_ids, attention_mask, **kwargs):
            prompts.append("".join(chr(int(token) - 100) for token in input_ids[0]))
            return torch.cat([input_ids, torch.tensor([[10 if len(prompts) == 1 else 11, 99]])], dim=1)

    class Model:
        slot_head = object() if slot_enabled else None
        backbone = Backbone()

        def __call__(self, **kwargs):
            model_calls.append(kwargs)
            return {"route_logits": torch.tensor([[-10.0, -10.0, 10.0]]),
                    "slot_logits": torch.tensor([[-10.0, 10.0]]) if slot_enabled else None}

    cls = mediq.measured_runtime_class({"mixed_answer_stage_token_budget": None})
    runtime = cls(tmp_path, max_input_tokens=None, max_new_tokens=None)
    runtime._loaded = True
    runtime.processor, runtime.tokenizer, runtime.model = Processor(), Tokenizer(), Model()
    runtime.slots, runtime.control_mode = ["duration", "onset"], control_mode
    runtime.calls, runtime.routing_calls, runtime.remaining = [], [], None
    original = [{"role": "user", "content": "cough"}]
    result = runtime.generate(original)
    assert result.route == ANSWER_AND_ASK_ROUTE
    assert result.completion == "answer question"
    assert result.slot == ("onset" if slot_enabled else None)
    assert len(runtime.calls) == 2
    assert len(runtime.routing_calls) == len(model_calls) == (2 if slot_enabled else 1)
    assert [call["input_tokens"] for call in runtime.routing_calls] == [
        call["input_ids"].shape[1] for call in model_calls
    ]
    if slot_enabled:
        assert model_calls[1]["slot_route_ids"].tolist() == [ROUTE_LABELS[ASK_ROUTE]]
        assert runtime.routing_calls[1]["input_tokens"] > runtime.routing_calls[0]["input_tokens"]
    if control_mode == "route_slot":
        assert "[ROUTE=ASK]" in prompts[1] and "[INQUIRY_SLOT=onset]" in prompts[1]
    elif control_mode == "none":
        assert all("[ROUTE=" not in prompt and "[INQUIRY_SLOT=" not in prompt for prompt in prompts)
    else:
        assert "[ROUTE=ANSWER_AND_ASK]" in prompts[1] and "[INQUIRY_SLOT=" not in prompts[1]
    assert original == [{"role": "user", "content": "cough"}]


@pytest.mark.parametrize("has_old_manifest", [False, True])
def test_main_refuses_to_append_legacy_protocol(monkeypatch, tmp_path, has_old_manifest):
    main, _ = evaluators(monkeypatch)
    rows = [{"example_id": str(index), "context": []} for index in range(3000)]
    monkeypatch.setattr(main, "read_rows", lambda path: rows)
    monkeypatch.setattr(main, "digest", lambda path: "current")
    output = tmp_path / "predictions.jsonl"
    original = '{"legacy":true}\n'
    output.write_text(original, encoding="utf-8")
    if has_old_manifest:
        output.with_suffix(".generation_manifest.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(output=output, checkpoint=tmp_path, label="test", max_samples=1, batch_size=4)
    with pytest.raises(RuntimeError, match="manifest"):
        main.generate(args)
    assert output.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "total_budget,answer_budget,expected_limits,remaining",
    [(128, None, [128, 108], 98), (None, 96, [96, None], None),
     (128, 96, [96, 108], 98), (None, None, [None, None], None)],
)
def test_measured_answer_stage_respects_optional_budgets(
    monkeypatch, tmp_path, total_budget, answer_budget, expected_limits, remaining,
):
    _, mediq = evaluators(monkeypatch)
    cls = mediq.measured_runtime_class({"mixed_answer_stage_token_budget": answer_budget})
    runtime = cls(tmp_path, max_input_tokens=None, max_new_tokens=None)
    runtime.calls, runtime.remaining, runtime.control_mode = [], total_budget, "none"
    limits = []

    def emit(messages, controls):
        limits.append(runtime.remaining)
        used = 20 if len(limits) == 1 else 10
        runtime.calls.append({"generated_tokens": used})
        if runtime.remaining is not None:
            runtime.remaining -= used
        return "answer" if len(limits) == 1 else "question"

    runtime.emit = emit
    runtime.predict_inquiry_slot = lambda messages: ("onset", 0.9)
    completion, slot, confidence = runtime._generate_answer_and_ask([{"role": "user", "content": "cough"}])
    assert completion == "answer question"
    assert (slot, confidence) == ("onset", 0.9)
    assert limits == expected_limits
    assert runtime.remaining == remaining
