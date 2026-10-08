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
        max_input_tokens: int = 2048,
        max_new_tokens: int = 96,
        route_threshold: float = 0.5,
        max_gpu_memory: str | None = None,
        max_cpu_memory: str = "48GiB",
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device_map = device_map
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
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
        if not self.slots:
            raise ValueError("Checkpoint has no inquiry-slot vocabulary")
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
            len(self.slots),
            manifest.get("route_loss_weight", 0.5),
            manifest.get("slot_loss_weight", 0.5),
            route_labels=self.route_labels,
            route_class_weights=manifest.get("route_class_weights"),
            slot_class_weights=manifest.get("slot_class_weights"),
            slot_names=self.slots,
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
        if heads.get("slot_class_weights") is not None:
            self.model.slot_class_weights.copy_(torch.tensor(heads["slot_class_weights"], dtype=torch.float32))
        self.model.eval()
        self._loaded = True
        return self

    def _input_device(self) -> torch.device:
        assert self.model is not None
        embeddings = self.model.backbone.get_input_embeddings()
        return embeddings.weight.device

    def _trim_context(self, ids: list[int], context_length: int, prompt_length: int) -> tuple[list[int], int]:
        if len(ids) <= self.max_input_tokens:
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
        return re.sub(r"\[INQUIRY_SLOT=[^\]]+\]", "", text).strip()

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
        controls = make_control_tokens(self.route_labels[route], slot if route == ASK_ROUTE else None)
        ids, context_length, prompt_length = encode_context_with_controls(self.processor, messages, controls)
        ids, _ = self._trim_context(ids, context_length, prompt_length)
        device = self._input_device()
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        output = self.model.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output[0, input_ids.shape[1]:]
        completion = self._clean_completion(self.tokenizer.decode(generated_ids, skip_special_tokens=True))
        return EvidenceAwareResult(
            route=route,
            slot=slot if route == ASK_ROUTE else None,
            completion=completion,
            ask_probability=ask_probability,
            route_confidence=route_confidence,
            slot_confidence=slot_confidence if route == ASK_ROUTE else None,
            route_probabilities=route_probabilities,
        )

