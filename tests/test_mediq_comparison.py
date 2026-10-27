import pytest

from proactive_consultation.mediq_comparison import (
    bootstrap_difference, canonical, minimal_edit_valid, parse_json,
    validate_judgment, visible_messages, wilson,
)


def test_hidden_fact_and_reference_are_not_generation_inputs():
    row = {"context": [{"role": "patient", "text": "我最近咳嗽。"}],
           "fact": "SECRET_DURATION", "target_question": "REFERENCE_QUESTION",
           "construction_plan": {"fact": {"value": "SECRET_DURATION"}}}
    visible = visible_messages(row)
    assert visible == [{"role": "user", "content": "我最近咳嗽。"}]
    assert "SECRET" not in str(visible) and "REFERENCE" not in str(visible)


def test_minimal_edit_rejects_rewrites_and_doctor_fact_deletion():
    pair = {"context_known": [{"role": "patient", "text": "咳嗽三天，夜间加重。"}],
            "context_missing": [{"role": "patient", "text": "咳嗽，夜间加重。"}],
            "construction_plan": {"deletions": [{"turn": 0, "quote": "三天"}]}}
    assert minimal_edit_valid(pair)
    pair["context_missing"][0]["text"] = "咳嗽。"
    assert not minimal_edit_valid(pair)
    pair["context_known"][0]["role"] = "doctor"
    pair["context_missing"][0] = {"role": "doctor", "text": "咳嗽，夜间加重。"}
    assert not minimal_edit_valid(pair)


def test_positive_target_question_requires_real_output_evidence():
    payload = {"missing_output": "咳嗽多久了？", "known_output": "有没有发热？"}
    judgment = {"missing_target_ask": True, "known_target_ask": False,
                "missing_quote": "咳嗽多久了？", "known_quote": ""}
    validate_judgment("paired", judgment, payload)
    judgment["known_target_ask"] = True
    judgment["known_quote"] = "咳嗽多久了？"
    with pytest.raises(ValueError, match="verbatim"):
        validate_judgment("paired", judgment, payload)


def test_string_boolean_does_not_silently_count_as_true():
    with pytest.raises(ValueError, match="boolean"):
        validate_judgment("audit", {"original_supported": "false", "missing_absent": True,
                                   "question_relevant": True}, {})


def test_paired_bootstrap_keeps_pair_correspondence():
    result = bootstrap_difference([0, 1, 0, 1], [0, 1, 0, 1])
    assert result["mean_difference"] == 0
    assert result["ci95"] == [0, 0]
    with pytest.raises(ValueError):
        bootstrap_difference([1, 0], [1])
    assert wilson([])["rate"] is None
    interval = wilson([True] * 10)
    assert interval["rate"] == 1 and interval["ci95"][0] < 1


def test_judge_json_can_be_extracted_without_accepting_non_objects():
    assert parse_json('```json\n{"valid":true}\n```') == {"valid": True}
    with pytest.raises(ValueError):
        parse_json("[]")


def test_batch_generation_counts_real_tokens_not_eos_padding(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import torch
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_mediq_comparison import generate_plain_batches
    from proactive_consultation import mediq_comparison as comparison
    protocol_dir = tmp_path / "protocol_fixture"
    protocol_dir.mkdir()
    (protocol_dir / "protocol.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(comparison, "OUT", protocol_dir)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 99
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [7] * len(text)}
        def decode(self, tokens, skip_special_tokens=True):
            return "".join({10: "a", 11: "b"}.get(int(t), "") for t in tokens)
    class Processor:
        tokenizer = Tokenizer()
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "".join(x["content"] for x in messages) + "X"
    class Backbone:
        def generate(self, input_ids, attention_mask, **kwargs):
            assert attention_mask[0, 0] == 0
            continuations = torch.tensor([[10, 99, 0], [10, 11, 99]])
            return torch.cat([input_ids, continuations], dim=1)
    runtime = SimpleNamespace(processor=Processor(), tokenizer=Tokenizer(),
        model=SimpleNamespace(backbone=Backbone()), _input_device=lambda: torch.device("cpu"),
        _trim_context=lambda ids, context_len, prompt_len: (ids, context_len),
        _clean_completion=lambda text: text)
    rows = [{"example_id": "short", "context": [{"role": "patient", "text": "A"}]},
            {"example_id": "long", "context": [{"role": "patient", "text": "ABC"}]}]
    generate_plain_batches(runtime, rows, [], "lora_e3", {"final_generation_token_budget": 128}, tmp_path, [])
    records = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text("utf8").splitlines()]
    assert [r["prediction"] for r in records] == ["a", "ab"]
    assert [r["generated_tokens"] for r in records] == [2, 3]
    assert [r["input_tokens"] for r in records] == [2, 4]
    assert all(r["generation_call_count"] == 1 and r["batch_size"] == 2 for r in records)
