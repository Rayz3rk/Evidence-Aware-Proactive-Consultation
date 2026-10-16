"""Check full-context encoding and EOS termination without a default output cap."""
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor

from proactive_consultation.evidence_aware_runtime import EvidenceAwareRuntime
from proactive_consultation.token_limits import generation_token_kwargs
from test_evidence_aware import DummyProcessor
from proactive_consultation.evidence_aware import build_examples


def test_unlimited_training_keeps_long_context_and_target():
    rows = [{"pair_id": "long", "task_type": "normal_dialogue",
             "messages": [{"role": "user", "content": "a" * 3000}],
             "completion": "b" * 700}]
    full = build_examples(DummyProcessor(), rows, {}, {})[0]
    zero = build_examples(DummyProcessor(), rows, {}, {}, max_length=0)[0]
    assert full == zero
    assert full["context_length"] > 3000
    assert len(full["input_ids"]) - full["prompt_length"] == 701


def test_unlimited_runtime_preserves_input_and_controls(tmp_path):
    runtime = EvidenceAwareRuntime(tmp_path)
    ids = list(range(4000))
    assert runtime._trim_context(ids, 3990, 4000) == (ids, 3990)
    zero = EvidenceAwareRuntime(tmp_path, max_input_tokens=0, max_new_tokens=0)
    assert zero.max_input_tokens is None and zero.max_new_tokens is None


class ForceEOSAfterThirtyTokens(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores.fill_(-float("inf"))
        scores[:, 2 if input_ids.shape[1] >= 31 else 3] = 0
        return scores


def test_generation_passes_old_default_cap_and_stops_at_eos():
    model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=64,
        n_embd=8, n_layer=1, n_head=1, bos_token_id=1, eos_token_id=2, pad_token_id=0))
    model.eval()
    # A saved checkpoint's old limit must also be overridden.
    model.generation_config.max_new_tokens = 4
    with torch.inference_mode():
        output = model.generate(input_ids=torch.tensor([[1]]), attention_mask=torch.ones(1, 1, dtype=torch.long),
            **generation_token_kwargs(), do_sample=False,
            logits_processor=[ForceEOSAfterThirtyTokens()])
    assert output.shape[1] == 32
    assert output[0, -1].item() == 2
