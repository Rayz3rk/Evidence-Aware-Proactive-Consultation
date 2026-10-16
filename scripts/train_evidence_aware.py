"""Train the evidence-aware proactive consultation model.

The checkpoint contains a LoRA adapter, a route head, a route-conditioned
inquiry-slot head, and a causal generator.  Audit-only construction fields
are used for labels and are never inserted into model-visible prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proactive_consultation.evidence_aware import (  # noqa: E402
    ANSWER_AND_ASK_ROUTE,
    ANSWER_ROUTE,
    ASK_ROUTE,
    EvidenceAwareCollator,
    EvidenceAwareDataset,
    EvidenceAwareModel,
    ROUTE_LABELS,
    add_control_tokens,
    build_examples,
    build_slot_vocabulary,
    load_slot_labels,
    route_for_row,
)
from proactive_consultation.token_limits import token_limit
from proactive_consultation.scope3000 import load_views, mix_training_rows, split_by_pair_id, task_counts  # noqa: E402


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def route_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(route_for_row(row) for row in rows)
    return {route: int(counts.get(route, 0)) for route in ROUTE_LABELS}


def inverse_sqrt_weights(labels: list[int], num_classes: int) -> list[float]:
    """Use mild inverse-frequency weighting without letting rare labels dominate."""
    counts = Counter(int(label) for label in labels)
    raw = [1.0 / math.sqrt(counts.get(index, 0)) if counts.get(index, 0) else 1.0 for index in range(num_classes)]
    mean = sum(raw) / max(1, len(raw))
    return [value / mean for value in raw]



class TorchStateDictTrainer(Trainer):
    """Save custom nn.Module checkpoints with torch.save.

    EvidenceAwareModel wraps a tied-weight language-model backbone but is not a
    Transformers PreTrainedModel.  The generic Trainer safetensors path rejects
    the tied input-embedding/lm-head tensors, whereas torch.save preserves the
    shared storage safely for our local checkpoints.
    """

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif self.data_collator is not None and hasattr(self.data_collator, "tokenizer") and self.data_collator.tokenizer is not None:
            self.data_collator.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

class LossHistoryCallback(TrainerCallback):
    """Persist train/eval loss events as JSONL while training is running."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": int(state.global_step), "epoch": state.epoch}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                record[key] = float(value)
            else:
                record[key] = value
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class StopAfterStepsCallback(TrainerCallback):
    """Stop a matched run without shortening its planned LR schedule."""

    def __init__(self, steps: int) -> None:
        self.steps = steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.steps:
            control.should_training_stop = True
        return control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-mode", choices=["joint", "normal_only", "question_only"], default="joint")
    parser.add_argument("--mix-ratio", default="2:1")
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--stop-after-steps", type=int, default=None,
                        help="Stop early while retaining the epochs/max-steps LR horizon")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--target-total", type=int, default=None, help="Target sampled training-instance count; requires a ratio such as 1:1")
    parser.add_argument("--quality-status", action="append", default=None, help="Allow this release quality_status; repeat for multiple statuses")
    parser.add_argument("--save-steps", type=int, default=0, help="Save Trainer checkpoints every N steps; 0 disables intermediate saves")
    parser.add_argument("--route-loss-weight", type=float, default=0.5)
    parser.add_argument("--slot-loss-weight", type=float, default=0.5)
    parser.add_argument("--control-mode", choices=["route_slot", "route_only", "slot_only", "none"], default="route_slot")
    parser.add_argument("--slot-condition", choices=["route", "context"], default="route")
    parser.add_argument("--disable-slot-head", action="store_true")
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None, help="No limit by default; 0 disables the limit")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--data-file", type=Path, default=ROOT / "data" / "EviGap-Med" / "sft_candidates.auto.jsonl")
    parser.add_argument("--audit-file", type=Path, default=ROOT / "data" / "EviGap-Med" / "paired_contexts.auto.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.route_loss_weight < 0 or args.slot_loss_weight < 0:
        parser.error("loss weights must be non-negative")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        parser.error("--stop-after-steps must be positive")

    set_seed(args.seed)
    config = load_config(args.config)
    data_file = resolve(ROOT, args.data_file).resolve()
    audit_file = resolve(ROOT, args.audit_file).resolve()
    model_path = resolve(ROOT, Path(config["local"]["model_path"])).resolve()
    output_dir = resolve(ROOT, args.output_dir).resolve() if args.output_dir else ROOT / "outputs" / "checkpoints" / f"evidence_aware_{args.task_mode}_{args.mix_ratio.replace(':', 'to')}_qwen3.5-0.8b-lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_statuses = set(args.quality_status or {"human_review_passed"})
    all_rows = load_views(data_file, args.task_mode, allowed_quality_statuses=quality_statuses)
    pair_slots = load_slot_labels(audit_file)
    missing = [row["pair_id"] for row in all_rows if row["task_type"] == "evidence_missing_question" and row["pair_id"] not in pair_slots]
    if missing:
        raise ValueError(f"Missing construction slots for inquiry rows: {missing[:5]}")
    slots = build_slot_vocabulary(all_rows, pair_slots)
    slot_to_id = {slot: index for index, slot in enumerate(slots)}
    train_rows, dev_rows = split_by_pair_id(all_rows, args.dev_fraction)
    pre_mix_train_rows = list(train_rows)
    if args.task_mode == "joint":
        train_rows = mix_training_rows(train_rows, args.mix_ratio, args.seed, target_total=args.target_total)
    if args.max_samples is not None:
        train_rows = train_rows[:args.max_samples]

    processor = AutoProcessor.from_pretrained(str(model_path))
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    control_tokens = add_control_tokens(tokenizer, slots)
    max_length = token_limit(args.max_length if args.max_length is not None else config["training"].get("max_length"))
    train_examples = build_examples(processor, train_rows, pair_slots, slot_to_id, max_length=max_length, control_mode=args.control_mode)
    dev_examples = build_examples(processor, dev_rows, pair_slots, slot_to_id, max_length=max_length, control_mode=args.control_mode)
    if not train_examples or not dev_examples:
        raise RuntimeError("Training or development split is empty")
    route_class_weights = inverse_sqrt_weights([example["route_label"] for example in train_examples], len(ROUTE_LABELS))
    slot_class_weights = inverse_sqrt_weights(
        [example["slot_label"] for example in train_examples if example["slot_label"] >= 0], len(slots)
    )

    manifest = {
        "format_version": 3,
        "architecture": "serial_route_conditioned_slot_v1",
        "decision_chain": "context -> route -> slot (ask only) -> conditioned generation",
        "generation_conditioning": "gold route/slot controls during teacher-forced training; predicted controls at inference",
        "control_mode": args.control_mode,
        "slot_condition": args.slot_condition,
        "slot_head_enabled": not args.disable_slot_head,
        "dataset": data_file.parent.name,
        "data_file": str(data_file),
        "audit_file_used_for_labels_only": str(audit_file),
        "model_path": os.path.relpath(model_path, output_dir),
        "model_path_absolute_at_train_time": str(model_path),
        "task_mode": args.task_mode,
        "allowed_quality_statuses": sorted(quality_statuses),
        "target_total": args.target_total,
        "mix_ratio": args.mix_ratio,
        "seed": args.seed,
        "max_length": max_length,
        "all_view_counts": task_counts(all_rows),
        "train_view_counts_before_mix": task_counts(pre_mix_train_rows),
        "train_view_counts_after_mix": task_counts(train_rows),
        "dev_view_counts": task_counts(dev_rows),
        "route_counts_all": route_counts(all_rows),
        "route_counts_train": route_counts(train_rows),
        "route_counts_dev": route_counts(dev_rows),
        "route_labels": ROUTE_LABELS,
        "route_definition": {
            ANSWER_ROUTE: "ordinary response without an explicit follow-up question",
            ASK_ROUTE: "proactive evidence-missing inquiry",
            ANSWER_AND_ASK_ROUTE: "ordinary response that also contains a follow-up question",
        },
        "slot_vocabulary": slots,
        "control_tokens": control_tokens,
        "route_loss_weight": args.route_loss_weight,
        "slot_loss_weight": args.slot_loss_weight,
        "route_class_weights": route_class_weights,
        "slot_class_weights": slot_class_weights,
        "deleted_fact_in_model_input": False,
        "audit_text_in_model_input": False,
        "label_policy": "normal_question_like_completions_are_answer_and_ask",
        "status": "prepared",
        "stop_after_steps": args.stop_after_steps,
    }
    (output_dir / "evidence_aware_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prepared_dir = output_dir / "prepared_data"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train_views.jsonl", train_rows), ("dev_views.jsonl", dev_rows)):
        with (prepared_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.dry_run:
        collator = EvidenceAwareCollator(tokenizer.pad_token_id)
        batch = collator(train_examples[: min(2, len(train_examples))])
        print(json.dumps({"manifest": manifest, "batch_shapes": {key: list(value.shape) for key, value in batch.items()}}, ensure_ascii=False, indent=2))
        return

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    backbone = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    backbone.config.use_cache = False
    lora_cfg = config["training"]["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    backbone = get_peft_model(backbone, peft_config)
    model = EvidenceAwareModel(
        backbone,
        0 if args.disable_slot_head else len(slots),
        args.route_loss_weight,
        args.slot_loss_weight,
        route_labels=ROUTE_LABELS,
        route_class_weights=route_class_weights,
        slot_class_weights=[] if args.disable_slot_head else slot_class_weights,
        slot_names=[] if args.disable_slot_head else slots,
        slot_condition=args.slot_condition,
    )
    if hasattr(model.backbone, "print_trainable_parameters"):
        model.backbone.print_trainable_parameters()

    train_dataset = EvidenceAwareDataset(train_examples)
    dev_dataset = EvidenceAwareDataset(dev_examples)
    collator = EvidenceAwareCollator(tokenizer.pad_token_id)
    train_cfg = config["training"]
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    per_device_batch_size = args.per_device_train_batch_size or train_cfg["per_device_train_batch_size"]
    gradient_accumulation_steps = args.gradient_accumulation_steps or train_cfg["gradient_accumulation_steps"]
    effective_batch = per_device_batch_size * gradient_accumulation_steps
    estimated_steps = max(1, math.ceil(len(train_dataset) / effective_batch * epochs))
    if args.max_steps > 0:
        estimated_steps = args.max_steps
    train_args_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type="linear",
        num_train_epochs=epochs,
        warmup_steps=max(1, math.floor(estimated_steps * train_cfg["warmup_ratio"])),
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=30,
        report_to="none",
        seed=args.seed,
        max_steps=args.max_steps,
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        tf32=train_cfg.get("tf32", False),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        optim=train_cfg.get("optim", "adamw_torch"),
        label_names=["labels", "route_labels", "slot_labels"],
    )
    if "eval_strategy" not in inspect.signature(TrainingArguments).parameters:
        train_args_kwargs["evaluation_strategy"] = train_args_kwargs.pop("eval_strategy")
    training_args = TrainingArguments(**train_args_kwargs)
    loss_log_path = output_dir / "loss_history.jsonl"
    if loss_log_path.exists():
        loss_log_path.unlink()
    callbacks = [LossHistoryCallback(loss_log_path)]
    if args.stop_after_steps is not None:
        callbacks.append(StopAfterStepsCallback(args.stop_after_steps))
    trainer = TorchStateDictTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset, data_collator=collator, callbacks=callbacks)
    trainer.train()

    # Persist the full wrapper state for exact ablation evaluation reloads.
    torch.save(model.state_dict(), output_dir / 'pytorch_model.bin')

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(str(adapter_dir), safe_serialization=False)
    processor.save_pretrained(str(output_dir))
    torch.save({
        "route_head": model.route_head.state_dict(),
        "route_embedding": model.route_embedding.state_dict(),
        "slot_head": model.slot_head.state_dict() if model.slot_head is not None else None,
        "num_slots": 0 if args.disable_slot_head else len(slots),
        "slot_names": [] if args.disable_slot_head else slots,
        "slot_condition": args.slot_condition,
        "route_labels": ROUTE_LABELS,
        "route_loss_weight": args.route_loss_weight,
        "slot_loss_weight": args.slot_loss_weight,
        "route_class_weights": route_class_weights,
        "slot_class_weights": [] if args.disable_slot_head else slot_class_weights,
    }, output_dir / "heads.pt")
    manifest.update({
        "status": "trained",
        "global_step": trainer.state.global_step,
        "epochs_completed": trainer.state.epoch,
        "data_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(audit_file.read_bytes()).hexdigest(),
        "loss_log": str(loss_log_path),
    })
    (output_dir / "evidence_aware_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved_evidence_aware_model={output_dir}")


if __name__ == "__main__":
    main()
