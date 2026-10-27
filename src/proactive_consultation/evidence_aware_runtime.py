"""Inference runtime for the evidence-aware consultation model."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from .token_limits import generation_token_kwargs, token_limit

from .evidence_aware import (
    ANSWER_AND_ASK_ROUTE,
    ANSWER_ROUTE,
    ASK_ROUTE,
    EvidenceAwareModel,
    ROUTE_LABELS,
    ROUTE_NAMES,
    encode_context_with_controls,
    make_control_tokens,
)


@dataclass(frozen=True)
class EvidenceAwareResult:
    """One displayed turn; slot describes an ASK or mixed-route follow-up."""

    route: str
    slot: str | None
    completion: str
    ask_probability: float
    route_confidence: float
    slot_confidence: float | None
    route_probabilities: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "slot": self.slot,
            "completion": self.completion,
            "ask_probability": self.ask_probability,
            "route_confidence": self.route_confidence,
            "slot_confidence": self.slot_confidence,
            "route_probabilities": self.route_probabilities,
        }


class EvidenceAwareRuntime:
    """Load one checkpoint and run route, slot, and generation inference."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device_map: str | None = "auto",
        max_input_tokens: int | None = None,
        max_new_tokens: int | None = None,
        route_threshold: float = 0.5,
        max_gpu_memory: str | None = None,
        max_cpu_memory: str = "48GiB",
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device_map = device_map
        self.max_input_tokens = token_limit(max_input_tokens)
        self.max_new_tokens = token_limit(max_new_tokens)
        self.route_threshold = float(route_threshold)
        self.max_gpu_memory = max_gpu_memory
        self.max_cpu_memory = max_cpu_memory
        if not 0.0 <= self.route_threshold <= 1.0:
            raise ValueError("route_threshold must be in [0, 1]")
        self.processor: Any | None = None
        self.tokenizer: Any | None = None
        self.model: EvidenceAwareModel | None = None
        self.slots: list[str] = []
        self.route_labels: dict[str, int] = dict(ROUTE_LABELS)
        self._loaded = False

    def load(self) -> "EvidenceAwareRuntime":
        if self._loaded:
            return self
        manifest_path = self.checkpoint / "evidence_aware_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing evidence-aware manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.slots = list(manifest.get("slot_vocabulary") or [])
        self.control_mode = str(manifest.get("control_mode", "route_slot"))
        self.slot_condition = str(manifest.get("slot_condition", "route"))
        slot_head_enabled = bool(manifest.get("slot_head_enabled", bool(self.slots)))
        if slot_head_enabled and not self.slots:
            raise ValueError("Checkpoint with a slot head has no inquiry-slot vocabulary")
        manifest_routes = manifest.get("route_labels") or ROUTE_LABELS
        self.route_labels = {str(name): int(index) for name, index in manifest_routes.items()}
        if ASK_ROUTE not in self.route_labels or ANSWER_ROUTE not in self.route_labels:
            raise ValueError("Checkpoint route vocabulary must include answer and ask")
        model_path = Path(manifest["model_path"]).expanduser()
        if not model_path.is_absolute():
            model_path = (self.checkpoint / model_path).resolve()
        adapter_path = self.checkpoint / "adapter"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Missing LoRA adapter: {adapter_path}")
        heads_path = self.checkpoint / "heads.pt"
        if not heads_path.exists():
            raise FileNotFoundError(f"Missing route/slot heads: {heads_path}")

        self.processor = AutoProcessor.from_pretrained(str(self.checkpoint))
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True,
        }
        if self.device_map is not None:
            load_kwargs["device_map"] = self.device_map
        if self.max_gpu_memory is not None:
            load_kwargs["max_memory"] = {0: self.max_gpu_memory, "cpu": self.max_cpu_memory}
        base = AutoModelForImageTextToText.from_pretrained(str(model_path), **load_kwargs)
        backbone = PeftModel.from_pretrained(base, str(adapter_path))
        self.model = EvidenceAwareModel(
            backbone,
            len(self.slots) if slot_head_enabled else 0,
            manifest.get("route_loss_weight", 0.5),
            manifest.get("slot_loss_weight", 0.5),
            route_labels=self.route_labels,
            route_class_weights=manifest.get("route_class_weights"),
            slot_class_weights=manifest.get("slot_class_weights") if slot_head_enabled else [],
            slot_names=self.slots if slot_head_enabled else [],
            slot_condition=self.slot_condition,
        )
        heads = torch.load(heads_path, map_location="cpu")
        self.model.route_head.load_state_dict(heads["route_head"])
        if heads.get("route_embedding") is None:
            raise RuntimeError(
                "This checkpoint predates serial route-conditioned slot inference; "
                "retrain it with scripts/train_evidence_aware.py."
            )
        self.model.route_embedding.load_state_dict(heads["route_embedding"])
        if self.model.slot_head is not None and heads.get("slot_head") is not None:
            self.model.slot_head.load_state_dict(heads["slot_head"])
        if heads.get("route_class_weights") is not None:
            self.model.route_class_weights.copy_(torch.tensor(heads["route_class_weights"], dtype=torch.float32))
        if self.model.slot_head is not None and heads.get("slot_class_weights") is not None:
            self.model.slot_class_weights.copy_(torch.tensor(heads["slot_class_weights"], dtype=torch.float32))
        self.model.eval()
        self._loaded = True
        return self

    def _input_device(self) -> torch.device:
        assert self.model is not None
        embeddings = self.model.backbone.get_input_embeddings()
        return embeddings.weight.device

    def _trim_context(self, ids: list[int], context_length: int, prompt_length: int) -> tuple[list[int], int]:
        if self.max_input_tokens is None or len(ids) <= self.max_input_tokens:
            return ids, context_length
        control_length = prompt_length - context_length
        if control_length + 1 > self.max_input_tokens:
            raise ValueError("max_input_tokens is too small for the internal control prefix")
        keep_context = self.max_input_tokens - control_length
        context_ids = ids[:context_length][-keep_context:]
        control_ids = ids[context_length:prompt_length]
        return context_ids + control_ids, len(context_ids)

    @staticmethod
    def _clean_completion(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
        text = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.I | re.S)
        for marker in ("[ROUTE=ANSWER]", "[ROUTE=ASK]", "[ROUTE=ANSWER_AND_ASK]"):
            text = text.replace(marker, "")
        text = re.sub(r"\[INQUIRY_SLOT=[^\]]+\]", "", text)
        return re.sub(r"\[GENERATION_STAGE=[^\]]+\]", "", text).strip()

    @torch.inference_mode()
    def _generate_completion(
        self,
        messages: Sequence[Mapping[str, str]],
        controls: Sequence[str],
    ) -> str:
        """Generate one completion from a message context and internal controls."""
        assert self.processor is not None and self.tokenizer is not None and self.model is not None
        ids, context_length, prompt_length = encode_context_with_controls(
            self.processor, messages, controls
        )
        ids, _ = self._trim_context(ids, context_length, prompt_length)
        device = self._input_device()
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        output = self.model.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_token_kwargs(self.max_new_tokens),
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output[0, input_ids.shape[1]:]
        return self._clean_completion(
            self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        )

    def _generate_answer_and_ask(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> tuple[str, str | None, float | None]:
        """Answer, select a slot from the augmented context, then ask.

        The follow-up uses the slot head learned on ASK views, conditioned on
        the ASK route.  This inference decomposition leaves the training
        targets unchanged and retains ANSWER_AND_ASK as the displayed route.
        """
        answer = self._generate_completion(
            messages,
            make_control_tokens(
                self.route_labels[ANSWER_ROUTE], None, self.control_mode
            ),
        )
        follow_up_messages = [dict(message) for message in messages]
        if answer:
            follow_up_messages.append({"role": "assistant", "content": answer})
        slot, slot_confidence = self.predict_inquiry_slot(follow_up_messages)
        follow_up_messages.append(
            {
                "role": "user",
                "content": (
                    "请基于以上回答，仅生成一个简短、自然的后续追问；"
                    "不要重复回答，不要解释你的生成过程。"
                ),
            }
        )
        question = self._generate_completion(
            follow_up_messages,
            self._inquiry_controls(slot),
        )
        completion = " ".join(part for part in (answer, question) if part).strip()
        return completion, slot, slot_confidence

    def _inquiry_controls(self, slot: str | None) -> list[str]:
        """Use ASK slot controls, with a no-slot mixed-route ablation fallback."""
        if slot is None:
            return make_control_tokens(
                self.route_labels[ANSWER_AND_ASK_ROUTE], None, self.control_mode
            )
        return make_control_tokens(self.route_labels[ASK_ROUTE], slot, self.control_mode)

    @torch.inference_mode()
    def predict_inquiry_slot(
        self, messages: Sequence[Mapping[str, str]]
    ) -> tuple[str | None, float | None]:
        """Select a follow-up slot from c+a using the ASK-conditioned head.

        A checkpoint with its slot head disabled returns no slot; it is a
        no-slot ablation rather than a slot-controlled mixed response.
        """
        self.load()
        assert self.processor is not None and self.model is not None
        if self.model.slot_head is None:
            return None, None
        context_ids, context_length, prompt_length = encode_context_with_controls(
            self.processor, messages
        )
        context_ids, context_length = self._trim_context(
            context_ids, context_length, prompt_length
        )
        device = self._input_device()
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            context_lengths=torch.tensor([context_length], dtype=torch.long, device=device),
            hard_route_condition=True,
            slot_route_ids=torch.tensor(
                [self.route_labels[ASK_ROUTE]], dtype=torch.long, device=device
            ),
        )
        slot_probs = torch.softmax(outputs["slot_logits"], dim=-1).detach().float().cpu()[0]
        slot_id = int(slot_probs.argmax())
        return self.slots[slot_id], float(slot_probs[slot_id])

    def _route_from_probs(self, probs: torch.Tensor) -> tuple[str, float, float]:
        names_by_id = {index: name for name, index in self.route_labels.items()}
        ask_id = self.route_labels[ASK_ROUTE]
        ask_probability = float(probs[ask_id])
        if ask_probability >= self.route_threshold:
            route_id = ask_id
        else:
            candidates = [index for name, index in self.route_labels.items() if name != ASK_ROUTE]
            route_id = max(candidates, key=lambda index: float(probs[index]))
        return names_by_id[route_id], ask_probability, float(probs[route_id])

    @torch.inference_mode()
    def predict_route_and_slot(
        self, messages: Sequence[Mapping[str, str]]
    ) -> tuple[str, str | None, float, float, float | None, dict[str, float]]:
        self.load()
        assert self.processor is not None and self.model is not None
        context_ids, context_length, prompt_length = encode_context_with_controls(self.processor, messages)
        context_ids, context_length = self._trim_context(context_ids, context_length, prompt_length)
        device = self._input_device()
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        context_lengths = torch.tensor([context_length], dtype=torch.long, device=device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_lengths=context_lengths,
            hard_route_condition=True,
        )
        route_probs_tensor = torch.softmax(outputs["route_logits"], dim=-1).detach().float().cpu()[0]
        route, ask_probability, route_confidence = self._route_from_probs(route_probs_tensor)
        route_probabilities = {
            name: float(route_probs_tensor[index])
            for name, index in self.route_labels.items()
        }
        slot: str | None = None
        slot_confidence: float | None = None
        if route == ASK_ROUTE and outputs.get("slot_logits") is not None:
            slot_probs = torch.softmax(outputs["slot_logits"], dim=-1).detach().float().cpu()[0]
            slot_id = int(slot_probs.argmax())
            slot = self.slots[slot_id]
            slot_confidence = float(slot_probs[slot_id])
        return route, slot, ask_probability, route_confidence, slot_confidence, route_probabilities

    @torch.inference_mode()
    def generate(self, messages: Sequence[Mapping[str, str]]) -> EvidenceAwareResult:
        self.load()
        assert self.processor is not None and self.tokenizer is not None and self.model is not None
        route, slot, ask_probability, route_confidence, slot_confidence, route_probabilities = self.predict_route_and_slot(messages)
        if route == ASK_ROUTE and slot is None:
            raise RuntimeError("Ask route was selected but no slot head is available")
        if route == ANSWER_AND_ASK_ROUTE:
            completion, slot, slot_confidence = self._generate_answer_and_ask(messages)
        else:
            controls = make_control_tokens(
                self.route_labels[route],
                slot if route == ASK_ROUTE else None,
                self.control_mode,
            )
            completion = self._generate_completion(messages, controls)
        return EvidenceAwareResult(
            route=route,
            slot=slot if route in {ASK_ROUTE, ANSWER_AND_ASK_ROUTE} else None,
            completion=completion,
            ask_probability=ask_probability,
            route_confidence=route_confidence,
            slot_confidence=slot_confidence if route in {ASK_ROUTE, ANSWER_AND_ASK_ROUTE} else None,
            route_probabilities=route_probabilities,
        )
