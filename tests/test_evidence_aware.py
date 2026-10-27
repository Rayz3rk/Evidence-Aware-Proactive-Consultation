import json
from types import SimpleNamespace

import torch
import pytest
from torch import nn

from proactive_consultation.evidence_aware import (
    ANSWER_ROUTE,
    ASK_ROUTE,
    ANSWER_AND_ASK_ROUTE,
    EvidenceAwareCollator,
    EvidenceAwareModel,
    build_examples,
    load_slot_labels,
    make_control_tokens,
    route_for_row,
    is_question_like,
    slot_token,
    ROUTE_LABELS,
)


class DummyTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1 + (ord(ch) % 47) for ch in str(text)]}


class DummyProcessor:
    tokenizer = DummyTokenizer()

    @staticmethod
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


class DummyBackbone(nn.Module):
    def __init__(self, hidden=8, vocab=128):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embed = nn.Embedding(vocab, hidden)
        self.lm = nn.Linear(hidden, vocab)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False, use_cache=False, **kwargs):
        hidden = self.embed(input_ids)
        return SimpleNamespace(logits=self.lm(hidden), hidden_states=(hidden,))


def test_slot_token_and_control_routes():
    assert slot_token("起病时间") == "[INQUIRY_SLOT=起病时间]"
    assert make_control_tokens(ROUTE_LABELS[ANSWER_ROUTE])
    assert make_control_tokens(ROUTE_LABELS[ASK_ROUTE], "onset")
    assert make_control_tokens(ROUTE_LABELS[ANSWER_AND_ASK_ROUTE])


def test_audit_slots_and_prompt_exclusion(tmp_path):
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps({
        "pair_id": "p1",
        "construction_plan": {"fact": {"slot": "onset", "value": "SECRET_VALUE", "evidence_quote": "SECRET_QUOTE"}},
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    slots = load_slot_labels(audit)
    assert slots == {"p1": "onset"}
    rows = [{
        "pair_id": "p1",
        "task_type": "evidence_missing_question",
        "messages": [{"role": "user", "content": "患者描述"}],
        "completion": "请问什么时候开始？",
    }]
    examples = build_examples(DummyProcessor(), rows, slots, {"onset": 0})
    assert examples[0]["task_type"] == "evidence_missing_question"
    assert examples[0]["slot_label"] == 0
    assert "SECRET" not in str(examples[0]["input_ids"])


def test_collator_masks_prompt_and_model_has_finite_loss_for_response_only_batch():
    rows = [{
        "pair_id": "p1",
        "task_type": "normal_dialogue",
        "messages": [{"role": "user", "content": "患者描述"}],
        "completion": "建议先观察。",
    }]
    examples = build_examples(DummyProcessor(), rows, {}, {"onset": 0})
    batch = EvidenceAwareCollator(0)(examples)
    assert torch.all(batch["labels"][0, : examples[0]["prompt_length"]] == -100)
    model = EvidenceAwareModel(DummyBackbone(), num_slots=1)
    output = model(**batch)
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["slot_loss"])


def test_joint_forward_has_route_and_slot_logits():
    rows = [
        {"pair_id": "p1", "task_type": "normal_dialogue", "messages": [{"role": "user", "content": "A"}], "completion": "答复"},
        {"pair_id": "p2", "task_type": "evidence_missing_question", "messages": [{"role": "user", "content": "B"}], "completion": "追问？"},
    ]
    examples = build_examples(DummyProcessor(), rows, {"p2": "onset"}, {"onset": 0})
    output = EvidenceAwareModel(DummyBackbone(), num_slots=1)(**EvidenceAwareCollator(0)(examples))
    assert output["route_logits"].shape == (2, 3)
    assert output["slot_logits"].shape == (2, 1)
    assert torch.isfinite(output["loss"])



def test_normal_dialogue_with_question_is_mixed_route():
    assert is_question_like("建议先观察，您是否发热？")
    assert route_for_row({"task_type": "normal_dialogue", "completion": "建议先观察。"}) == ANSWER_ROUTE
    assert route_for_row({"task_type": "normal_dialogue", "completion": "建议先观察，您是否发热？"}) == ANSWER_AND_ASK_ROUTE
    assert route_for_row({"task_type": "evidence_missing_question", "completion": "请问什么时候开始？"}) == ASK_ROUTE


def test_plain_lora_configuration_has_generation_only_gradients():
    rows = [{"pair_id": "p1", "task_type": "normal_dialogue",
             "messages": [{"role": "user", "content": "患者描述"}], "completion": "建议先观察。"}]
    examples = build_examples(DummyProcessor(), rows, {}, {}, control_mode="none")
    model = EvidenceAwareModel(DummyBackbone(), num_slots=0, route_loss_weight=0, slot_loss_weight=0)
    output = model(**EvidenceAwareCollator(0)(examples))
    assert torch.equal(output["loss"], output["lm_loss"])
    output["loss"].backward()
    assert model.backbone.embed.weight.grad.abs().sum() > 0
    assert model.route_head.weight.grad is None or torch.count_nonzero(model.route_head.weight.grad) == 0
    assert model.route_embedding.weight.grad is None


def test_slot_route_override_reuses_ask_embedding_without_relabeling_route():
    model = EvidenceAwareModel(DummyBackbone(), num_slots=2)
    with torch.no_grad():
        model.route_head.weight.zero_()
        model.route_head.bias.copy_(torch.tensor([8.0, 0.0, 0.0]))
        model.route_embedding.weight.zero_()
        model.route_embedding.weight[ROUTE_LABELS[ASK_ROUTE], 0] = 3.0
        model.slot_head.weight.zero_()
        model.slot_head.bias.zero_()
        model.slot_head.weight[1, model.hidden_size] = 1.0
    batch = {"input_ids": torch.tensor([[1, 2]]), "context_lengths": torch.tensor([2])}
    ordinary = model(**batch, hard_route_condition=True)
    follow_up = model(
        **batch, hard_route_condition=True,
        slot_route_ids=torch.tensor([ROUTE_LABELS[ASK_ROUTE]]),
    )
    assert ordinary["route_logits"].argmax(-1).item() == ROUTE_LABELS[ANSWER_ROUTE]
    assert torch.equal(ordinary["route_logits"], follow_up["route_logits"])
    assert ordinary["slot_logits"][0, 1].item() == 0.0
    assert follow_up["slot_logits"][0, 1].item() == 3.0
    assert follow_up["slot_condition_route"].item() == ROUTE_LABELS[ASK_ROUTE]
    with pytest.raises(ValueError, match="one route id per context"):
        model(**batch, slot_route_ids=torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="unknown route id"):
        model(**batch, slot_route_ids=torch.tensor([99]))


def test_mixed_training_views_keep_slot_labels_masked():
    rows = [{
        "pair_id": "p1", "task_type": "normal_dialogue",
        "messages": [{"role": "user", "content": "Cough for three days"}],
        "completion": "Please rest. When is the cough worse?",
    }]
    example = build_examples(
        DummyProcessor(), rows, {"p1": "aggravating_factor"}, {"aggravating_factor": 0}
    )[0]
    assert example["route"] == ANSWER_AND_ASK_ROUTE
    assert example["slot_label"] == -100
