"""Prepare, generate, judge and summarize the supplementary MediQ comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from proactive_consultation import mediq_comparison as m
from proactive_consultation.token_limits import generation_token_kwargs


def judgment_records(stage: str, judge: str) -> list[dict]:
    records = []
    for path in sorted(m.OUT.glob(f"{stage}.{judge}.*.judgments.jsonl")):
        records.extend(m.read_rows(path))
    return records


def load_credentials() -> tuple[str, str]:
    values = dict(os.environ)
    dotenv = ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text("utf8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                values.setdefault(k.strip(), v.strip().strip("\"'"))
    key = values.get("JUDGE_API_KEY", "")
    if not key:
        raise RuntimeError("Missing JUDGE_API_KEY")
    return values.get("JUDGE_BASE_URL", "").rstrip("/"), key


def judge(args: argparse.Namespace) -> None:
    protocol = m.prepare()
    base, key = load_credentials()
    stage = args.stage
    judge_model = protocol["primary_requested_judge"] if stage == "audit" else args.judge
    if stage != "audit":
        m.scoring_protocol()
    shard = args.method or "all"
    path = m.OUT / ("audit.judgments.jsonl" if stage == "audit" else f"{stage}.{judge_model}.{shard}.judgments.jsonl")
    old = m.read_rows(path) if stage == "audit" else judgment_records(stage, judge_model)
    protocol_hash = m.digest(m.OUT / "protocol.json")
    jobs = []
    pairs = m.read_rows(m.OUT / "pair_candidates.jsonl")
    if stage == "audit":
        for p in pairs:
            jobs.append((p["pair_id"], None, {
                "original_context": p["context_known"], "missing_context": p["context_missing"],
                "fact": p["construction_plan"]["fact"], "probe_question": p["construction_plan"]["probe_question"]}))
    else:
        rows = m.prediction_rows()
        selected_methods = (args.method,) if args.method else m.METHODS
        for method in selected_methods:
            predictions = m.read_rows(m.OUT / method / "predictions.jsonl")
            if [r["example_id"] for r in predictions] != [r["example_id"] for r in rows]:
                raise RuntimeError(f"Incomplete or mismatched predictions for {method}")
            indexed = {r["example_id"]: r for r in predictions}
            if stage == "natural":
                for r in rows:
                    if r["cohort"] == "natural":
                        jobs.append((r["example_id"], method, {"context": r["context"], "candidate": indexed[r["example_id"]]["prediction"]}))
            else:
                for p in pairs:
                    pid = p["pair_id"]
                    if pid + ":known" in indexed:
                        jobs.append((pid, method, {
                            "original_context": p["context_known"], "missing_context": p["context_missing"],
                            "fact": p["construction_plan"]["fact"], "probe_question": p["construction_plan"]["probe_question"],
                            "known_output": indexed[pid + ":known"]["prediction"],
                            "missing_output": indexed[pid + ":missing"]["prediction"]}))
    prompt = {"audit": m.AUDIT_PROMPT, "natural": m.NATURAL_PROMPT, "paired": m.PAIRED_PROMPT}[stage]
    def job_hash(payload):
        import hashlib
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    expected = {(identifier, method): job_hash(payload) for identifier, method, payload in jobs}
    done = set()
    for record in old:
        identity = (record["id"], record["method"])
        if stage != "audit" and args.method and record["method"] != args.method:
            continue
        if identity not in expected or record["input_sha256"] != expected[identity] or record["protocol_sha256"] != protocol_hash:
            raise RuntimeError("Judge resume input/protocol mismatch")
        if record["requested_model"] != judge_model:
            raise RuntimeError("Judge model changed")
        if stage != "audit" and record.get("judge_protocol_sha256") != m.digest(m.OUT / "judge_protocol.json"):
            raise RuntimeError("Scoring protocol changed")
        if record["valid"]:
            done.add(identity)
    def call(job):
        identifier, method, payload = job
        record = dict(id=identifier, method=method, input_sha256=job_hash(payload),
                      protocol_sha256=protocol_hash, requested_model=judge_model, valid=False)
        if stage != "audit":
            record["judge_protocol_sha256"] = m.digest(m.OUT / "judge_protocol.json")
        body = {"model": judge_model, "temperature": 0,
                "max_tokens": 700, "messages": [{"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
        for attempt in range(3):
            try:
                request = urllib.request.Request(base + "/chat/completions",
                    data=json.dumps(body).encode("utf8"), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
                started = time.perf_counter()
                with urllib.request.urlopen(request, timeout=120) as response:
                    obj = json.load(response)
                raw = obj["choices"][0]["message"].get("content") or ""
                record.update(raw_response=raw, served_model=obj.get("model"), usage=obj.get("usage"),
                              latency_seconds=time.perf_counter() - started, attempts=attempt + 1)
                result = m.parse_json(raw)
                record["parsed_response"] = result
                m.validate_judgment(stage, result, payload)
                record.update(valid=True, judgment=result, raw_response=raw, served_model=obj.get("model"),
                              usage=obj.get("usage"), latency_seconds=time.perf_counter() - started,
                              attempts=attempt + 1)
                return record
            except Exception as error:
                # Never log credentials, headers, request objects, or response bodies on failure.
                record.update(error_type=type(error).__name__, http_status=getattr(error, "code", None), attempts=attempt + 1)
                if isinstance(error, ValueError):
                    record["validation_error"] = str(error)
                if getattr(error, "code", None) in {401, 403, 404, 410}:
                    break
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
        return record
    pending = [job for job in jobs if (job[0], job[1]) not in done]
    print(json.dumps(dict(stage=stage, judge=judge_model, total=len(jobs), pending=len(pending))), flush=True)
    with path.open("a", encoding="utf8") as handle, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(call, job) for job in pending]
        failures = 0
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            failures += not result["valid"]
            if index % 10 == 0 or index == len(futures):
                print(json.dumps(dict(stage=stage, completed=len(done) + index, total=len(jobs), failures=failures)), flush=True)
    if failures:
        raise RuntimeError(f"{failures} invalid judgments; resume to retry before reporting")


def generate_plain_batches(runtime, rows: list[dict], done: list[dict], method: str,
                           protocol: dict, output_dir: Path, digit_ids: list[int]) -> None:
    """Batch only the prompt-driven decoders; retain per-view stage traces."""
    import torch
    from proactive_consultation.evidence_aware import encode_context_with_controls
    batch_size = 8
    def emit_many(message_sets, *, budget, controls=(), allowed=None):
        sequences = []
        for messages in message_sets:
            ids, context_len, prompt_len = encode_context_with_controls(runtime.processor, messages, controls)
            ids, _ = runtime._trim_context(ids, context_len, prompt_len)
            sequences.append(ids)
        width = max(len(ids) for ids in sequences)
        pad = runtime.tokenizer.pad_token_id
        inputs = torch.tensor([[pad] * (width - len(ids)) + ids for ids in sequences],
                              device=runtime._input_device(), dtype=torch.long)
        masks = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in sequences],
                             device=inputs.device, dtype=torch.long)
        options = {"prefix_allowed_tokens_fn": lambda batch_id, input_ids: allowed} if allowed is not None else {}
        generation_options = generation_token_kwargs(budget)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = runtime.model.backbone.generate(input_ids=inputs, attention_mask=masks,
                **generation_options, do_sample=False, use_cache=True,
                pad_token_id=pad, eos_token_id=runtime.tokenizer.eos_token_id, **options)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        texts, traces = [], []
        for index, ids in enumerate(sequences):
            tokens = generated[index, width:]
            hits = (tokens == runtime.tokenizer.eos_token_id).nonzero()
            if len(hits):
                tokens = tokens[:int(hits[0].item()) + 1]
            raw = runtime.tokenizer.decode(tokens, skip_special_tokens=True)
            texts.append(runtime._clean_completion(raw))
            traces.append(dict(input_tokens=len(ids), generated_tokens=len(tokens), raw_output=raw,
                token_cap=budget, intentional_one_token_choice=allowed is not None,
                hit_token_cap=allowed is not None and len(tokens) == budget and
                    (not len(tokens) or int(tokens[-1]) != runtime.tokenizer.eos_token_id),
                latency_seconds=elapsed, batch_size=len(sequences), batch_wall_seconds=elapsed))
        return texts, traces
    # Regenerate a partial batch on resume to preserve its original padding.
    start = (len(done) // batch_size) * batch_size
    with (output_dir / "predictions.jsonl").open("a", encoding="utf8") as handle:
        for offset in range(start, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            message_sets = [m.visible_messages(row) for row in batch]
            started = time.perf_counter()
            calls = [[] for _ in batch]
            extras = [{} for _ in batch]
            if method == "lora_e3":
                predictions, traces = emit_many(message_sets, budget=protocol["final_generation_token_budget"])
                for i, trace in enumerate(traces):
                    calls[i].append(trace)
            else:
                rationale_messages = [[{"role": "system", "content": m.RG_PROMPT}] + messages for messages in message_sets]
                rationales, traces = emit_many(rationale_messages, budget=protocol["mediq_rationale_token_budget"])
                for i, trace in enumerate(traces):
                    calls[i].append(trace)
                score_messages = [[{"role": "system", "content": m.SCALE_PROMPT}] + messages + [
                    {"role": "user", "content": "内部参考分析（不是患者事实）：" + reason}]
                    for messages, reason in zip(message_sets, rationales)]
                score_texts, traces = emit_many(score_messages, budget=None, controls=["置信度等级："], allowed=digit_ids)
                scores = [int(next(ch for ch in text if ch in "12345")) for text in score_texts]
                response_messages = []
                for i, (messages, reason, score) in enumerate(zip(message_sets, rationales, scores)):
                    calls[i].append(traces[i])
                    route = "answer" if score >= protocol["confidence_threshold"] else "ask"
                    prompt = m.ANSWER_PROMPT if route == "answer" else m.ASK_PROMPT
                    response_messages.append([{"role": "system", "content": prompt}] + messages + [
                        {"role": "user", "content": f"内部参考分析：{reason}\n置信度等级：{score}\n请输出下一轮医生话语。"}])
                    extras[i] = dict(route=route, confidence_level=score, rationale=reason)
                predictions, traces = emit_many(response_messages, budget=protocol["final_generation_token_budget"])
                for i, trace in enumerate(traces):
                    calls[i].append(trace)
            elapsed = time.perf_counter() - started
            records = []
            for i, row in enumerate(batch):
                record = dict(row, prediction=predictions[i], **extras[i], generation_calls=calls[i],
                    generation_call_count=len(calls[i]), latency_seconds=elapsed, latency_basis="batched_request_wall_time",
                    batch_size=len(batch), generated_tokens=sum(c["generated_tokens"] for c in calls[i]),
                    input_tokens=sum(c["input_tokens"] for c in calls[i]),
                    any_stage_hit_token_cap=any(c["hit_token_cap"] for c in calls[i]),
                    protocol_sha256=m.digest(m.OUT / "protocol.json"))
                if offset + i < len(done):
                    if record["prediction"] != done[offset + i]["prediction"]:
                        raise RuntimeError("Partial-batch regeneration changed a stored prediction")
                else:
                    records.append(record)
            handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
            handle.flush()
            completed = offset + len(batch)
            m.write_json(output_dir / "status.json", dict(method=method, completed=completed,
                         total=len(rows), status="complete" if completed == len(rows) else "running"))
            print(f"{method}: {completed}/{len(rows)}", flush=True)


def measured_runtime_class(protocol: dict):
    """Build the measured decoder while keeping optional model imports lazy."""
    import torch
    from proactive_consultation.evidence_aware_runtime import EvidenceAwareRuntime
    from proactive_consultation.evidence_aware import encode_context_with_controls
    class MeasuredRuntime(EvidenceAwareRuntime):
        calls: list[dict]
        routing_calls: list[dict]
        remaining: int | None

        def _record_routing_input(self, messages, stage):
            ids, context_length, prompt_length = encode_context_with_controls(self.processor, messages)
            ids, _ = self._trim_context(ids, context_length, prompt_length)
            self.routing_calls.append(dict(stage=stage, input_tokens=len(ids)))

        def predict_route_and_slot(self, messages):
            self._record_routing_input(messages, "initial_route")
            return super().predict_route_and_slot(messages)

        def predict_inquiry_slot(self, messages):
            if self.model.slot_head is not None:
                self._record_routing_input(messages, "follow_up_slot")
            return super().predict_inquiry_slot(messages)

        def emit(self, messages, controls=(), *, budget=None, allowed=None):
            ids, context_length, prompt_length = encode_context_with_controls(self.processor, messages, controls)
            ids, _ = self._trim_context(ids, context_length, prompt_length)
            inputs = torch.tensor([ids], dtype=torch.long, device=self._input_device())
            limit = budget if budget is not None else self.remaining
            if limit is not None and limit <= 0:
                return ""
            options = {}
            if allowed is not None:
                options["prefix_allowed_tokens_fn"] = lambda batch_id, input_ids: allowed
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            generation_options = generation_token_kwargs(limit)
            with torch.inference_mode():
                out = self.model.backbone.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    do_sample=False, use_cache=True, **generation_options,
                    pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, **options)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            tokens = out[0, inputs.shape[1]:]
            raw = self.tokenizer.decode(tokens, skip_special_tokens=True)
            eos = self.tokenizer.eos_token_id
            self.calls.append(dict(input_tokens=len(ids), generated_tokens=len(tokens), raw_output=raw,
                                   token_cap=limit, intentional_one_token_choice=allowed is not None,
                                   hit_token_cap=limit is not None and allowed is None and len(tokens) == limit and (not len(tokens) or int(tokens[-1]) != eos),
                                   latency_seconds=time.perf_counter() - started))
            if budget is None and self.remaining is not None:
                self.remaining -= len(tokens)
            return self._clean_completion(raw)
        def _generate_completion(self, messages, controls):
            return self.emit(messages, controls)
        def _generate_answer_and_ask(self, messages):
            from proactive_consultation.evidence_aware import ANSWER_ROUTE, make_control_tokens
            saved = self.remaining
            stage_budget = protocol["mixed_answer_stage_token_budget"]
            budgets = [budget for budget in (saved, stage_budget) if budget is not None]
            self.remaining = min(budgets) if budgets else None
            first_answer_call = len(self.calls)
            answer = self.emit(messages, make_control_tokens(self.route_labels[ANSWER_ROUTE], None, self.control_mode))
            used = sum(call["generated_tokens"] for call in self.calls[first_answer_call:])
            self.remaining = max(saved - used, 0) if saved is not None else None
            slot_context = [dict(message) for message in messages]
            if answer:
                slot_context.append({"role": "assistant", "content": answer})
            slot, slot_confidence = self.predict_inquiry_slot(slot_context)
            follow = slot_context + [{"role": "user", "content":
                "请基于以上回答，仅生成一个简短、自然的后续追问；不要重复回答，不要解释你的生成过程。"}]
            question = self.emit(follow, self._inquiry_controls(slot))
            return " ".join(v for v in (answer, question) if v).strip(), slot, slot_confidence
    return MeasuredRuntime


def generate(args: argparse.Namespace) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(4)
    from proactive_consultation.evidence_aware_runtime import EvidenceAwareRuntime
    from proactive_consultation.evidence_aware import encode_context_with_controls
    protocol = m.prepare()
    rows = m.prediction_rows()
    if args.limit:
        rows = rows[:args.limit]
    method = args.method
    output_dir = m.OUT / method if not args.limit else m.OUT / "smoke" / f"{method}_{args.limit}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.jsonl"
    manifest = dict(protocol_sha256=m.digest(m.OUT / "protocol.json"), audit_sha256=m.digest(m.OUT / "audit.judgments.jsonl"),
                    evaluator_sha256=m.digest(Path(__file__)), method=method,
                    runtime_sha256=m.digest(ROOT / "src/proactive_consultation/evidence_aware_runtime.py"),
                    model_sha256=m.digest(ROOT / "src/proactive_consultation/evidence_aware.py"),
                    answer_and_ask_generation="answer_then_ask_with_predicted_slot_v2",
                    generation_batch_size=1 if method == "easc_e3" else 8,
                    example_ids=[r["example_id"] for r in rows])
    checkpoint = m.EASC if method == "easc_e3" else m.PLAIN
    manifest["adapter_sha256"] = m.digest(checkpoint / "adapter/adapter_model.bin")
    if method == "easc_e3":
        manifest["state_sha256"] = m.digest(checkpoint / "checkpoint-168/pytorch_model.bin")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text("utf8")) != manifest:
        raise RuntimeError("Generation resume manifest mismatch")
    if not manifest_path.exists() and output_path.exists() and output_path.stat().st_size:
        raise RuntimeError("Existing predictions have no generation manifest; refusing to mix protocols")
    m.write_json(manifest_path, manifest)
    done = m.read_rows(output_path)
    if len(done) > len(rows) or any(a["example_id"] != b["example_id"] or a["context"] != b["context"] for a, b in zip(done, rows)):
        raise RuntimeError("Generation resume prefix mismatch")
    if len(done) == len(rows):
        print("already complete", flush=True)
        return
    MeasuredRuntime = measured_runtime_class(protocol)
    runtime = MeasuredRuntime(checkpoint, max_input_tokens=protocol["max_input_tokens_per_call"],
                              max_new_tokens=protocol["final_generation_token_budget"], max_gpu_memory="5GiB")
    runtime.load()
    if method == "easc_e3":
        state = torch.load(checkpoint / "checkpoint-168/pytorch_model.bin", map_location="cpu", weights_only=True)
        runtime.model.load_state_dict(state, strict=True)
        del state
    else:
        plain_manifest = json.loads((checkpoint / "evidence_aware_manifest.json").read_text("utf8"))
        if (plain_manifest["global_step"] != 168 or plain_manifest["route_loss_weight"] != 0
                or plain_manifest["slot_loss_weight"] != 0 or plain_manifest["control_mode"] != "none"):
            raise RuntimeError("Ordinary LoRA training is incomplete or not matched")
        for split in ("train", "dev"):
            a = m.read_rows(checkpoint / f"prepared_data/{split}_views.jsonl")
            b = m.read_rows(m.EASC / f"prepared_data/{split}_views.jsonl")
            if a != b:
                raise RuntimeError("LoRA and EASC training views differ")
    runtime.model.eval()
    digits = [runtime.tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 6)]
    if any(len(x) != 1 for x in digits):
        raise RuntimeError("Constrained confidence digits must be single tokens")
    digit_ids = [x[0] for x in digits]
    if method != "easc_e3":
        generate_plain_batches(runtime, rows, done, method, protocol, output_dir, digit_ids)
        return
    with output_path.open("a", encoding="utf8") as handle:
        for index, row in enumerate(rows[len(done):], len(done)):
            runtime.calls, runtime.remaining = [], protocol["final_generation_token_budget"]
            runtime.routing_calls = []
            messages = m.visible_messages(row)
            started = time.perf_counter()
            extras = {}
            if method == "easc_e3":
                result = runtime.generate(messages)
                prediction = result.completion
                extras = {k: v for k, v in result.as_dict().items() if k != "completion"}
                extras.update(routing_forward_passes=len(runtime.routing_calls),
                              routing_input_tokens=sum(call["input_tokens"] for call in runtime.routing_calls),
                              routing_calls=runtime.routing_calls)
            elif method == "lora_e3":
                prediction = runtime.emit(messages)
            else:
                rationale = runtime.emit([{"role": "system", "content": m.RG_PROMPT}] + messages,
                                         budget=protocol["mediq_rationale_token_budget"])
                score_messages = [{"role": "system", "content": m.SCALE_PROMPT}] + messages + [
                    {"role": "user", "content": "内部参考分析（不是患者事实）：" + rationale}]
                score_text = runtime.emit(score_messages, ["置信度等级："], budget=None, allowed=digit_ids)
                score = int(next(ch for ch in score_text if ch in "12345"))
                route = "answer" if score >= protocol["confidence_threshold"] else "ask"
                prompt = m.ANSWER_PROMPT if route == "answer" else m.ASK_PROMPT
                response_messages = [{"role": "system", "content": prompt}] + messages + [
                    {"role": "user", "content": f"内部参考分析：{rationale}\n置信度等级：{score}\n请输出下一轮医生话语。"}]
                prediction = runtime.emit(response_messages)
                extras = dict(route=route, confidence_level=score, rationale=rationale)
            record = dict(row, prediction=prediction, **extras, generation_calls=runtime.calls,
                          generation_call_count=len(runtime.calls), latency_seconds=time.perf_counter() - started,
                          generated_tokens=sum(c["generated_tokens"] for c in runtime.calls),
                          input_tokens=sum(c["input_tokens"] for c in runtime.calls) + extras.get("routing_input_tokens", 0),
                          any_stage_hit_token_cap=any(c["hit_token_cap"] for c in runtime.calls),
                          protocol_sha256=manifest["protocol_sha256"])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            m.write_json(output_dir / "status.json", dict(method=method, completed=index + 1, total=len(rows),
                         status="complete" if index + 1 == len(rows) else "running"))
            if (index + 1) % 10 == 0:
                print(f"{method}: {index + 1}/{len(rows)}", flush=True)


def summarize() -> None:
    protocol = m.prepare()
    predictions = {method: m.read_rows(m.OUT / method / "predictions.jsonl") for method in m.METHODS}
    expected_rows = m.prediction_rows()
    for method, rows in predictions.items():
        if [r["example_id"] for r in rows] != [r["example_id"] for r in expected_rows]:
            raise RuntimeError(f"Incomplete predictions: {method}")
    scoring = m.scoring_protocol()
    judgments = {}
    for stage in ("natural", "paired"):
        # Last successful judgment wins; failed retries never erase a valid rating.
        judgments[stage] = {(r["id"], r["method"]): r["judgment"] for r in judgment_records(stage, m.JUDGES[0]) if r["valid"]}
    report = {"protocol": protocol, "judge_protocol": scoring, "primary_judge": m.JUDGES[0],
              "methods": {}, "differences_easc_minus_mediq_style": {}}
    pair_metadata = {r["pair_id"]: r for r in m.read_rows(m.OUT / "pair_candidates.jsonl")}
    natural_metadata = {r["example_id"]: r for r in expected_rows if r["cohort"] == "natural"}
    original_main_ids = {r["example_id"] for r in m.read_rows(ROOT / "outputs/evaluation/serial_main_1000/serial_e3.predictions.jsonl")}
    for stage in ("natural", "paired"):
        ids_by_method = {method: {i for i, name in judgments[stage] if name == method} for method in m.METHODS}
        common = set.intersection(*ids_by_method.values())
        expected = protocol["natural_n"] if stage == "natural" else (len(expected_rows) - protocol["natural_n"]) // 2
        if len(common) != expected:
            raise RuntimeError(f"Incomplete common {stage} judgments: {len(common)}/{expected}")
        ids = sorted(common)
        for method in m.METHODS:
            values = [judgments[stage][(i, method)] for i in ids]
            summary = {"n": len(values)}
            if stage == "natural":
                for key in ("relevance", "information_gain", "non_redundancy", "medical_safety", "overall"):
                    summary[key] = sum(v[key] for v in values) / len(values)
                for key in ("contains_useful_question", "redundant_question", "unsafe"):
                    summary[key] = m.wilson([v[key] for v in values])
            else:
                summary["missing_target_ask"] = m.wilson([v["missing_target_ask"] for v in values])
                summary["known_target_ask"] = m.wilson([v["known_target_ask"] for v in values])
                summary["paired_success"] = m.wilson([v["missing_target_ask"] and not v["known_target_ask"] for v in values])
            report["methods"].setdefault(method, {})[stage] = summary
            source = lambda i: natural_metadata[i]["source_dataset"] if stage == "natural" else pair_metadata[i]["source"]
            summary["by_source"] = {name: m.summarize_judgments(stage,
                [judgments[stage][(i, method)] for i in ids if source(i) == name]) for name in sorted({source(i) for i in ids})}
            if stage == "natural":
                summary["disjoint_from_main_1000"] = m.summarize_judgments(stage,
                    [judgments[stage][(i, method)] for i in ids if i not in original_main_ids])
            else:
                slot = lambda i: pair_metadata[i]["construction_plan"]["fact"]["slot"]
                summary["by_slot"] = {name: m.summarize_judgments(stage,
                    [judgments[stage][(i, method)] for i in ids if slot(i) == name]) for name in sorted({slot(i) for i in ids})}
        keys = ("overall", "contains_useful_question") if stage == "natural" else ("missing_target_ask", "known_target_ask", "paired_success")
        for key in keys:
            def value(i, method):
                r = judgments[stage][(i, method)]
                return r["missing_target_ask"] and not r["known_target_ask"] if key == "paired_success" else r[key]
            report["differences_easc_minus_mediq_style"][stage + ":" + key] = m.bootstrap_difference(
                [value(i, "easc_e3") for i in ids], [value(i, "mediq_style_e3") for i in ids])
    for method, rows in predictions.items():
        report["methods"][method]["cost"] = {
            "n": len(rows), **{k: sum(r[k] for r in rows) / len(rows) for k in
                ("generation_call_count", "generated_tokens", "input_tokens", "latency_seconds", "any_stage_hit_token_cap")}}
        report["methods"][method]["predicted_routes"] = dict(__import__("collections").Counter(r.get("route", "uncontrolled") for r in rows))
        report["methods"][method]["confidence_levels"] = dict(__import__("collections").Counter(r.get("confidence_level") for r in rows if "confidence_level" in r))
    audit = {r["id"]: r["judgment"] for r in m.read_rows(m.OUT / "audit.judgments.jsonl") if r["valid"]}
    report["audit"] = {"candidates": len(audit), "accepted": (len(expected_rows) - protocol["natural_n"]) // 2,
        "accepted_sources": dict(__import__("collections").Counter(p["source"] for p in m.read_rows(m.OUT / "pair_candidates.jsonl")
            if p["pair_id"] in audit and all(audit[p["pair_id"]][k] for k in ("original_supported", "missing_absent", "question_relevant"))))}
    secondary = {}
    secondary_complete = True
    for stage in ("natural", "paired"):
        records = {(r["id"], r["method"]): r["judgment"] for r in judgment_records(stage, m.JUDGES[1]) if r["valid"]}
        count = protocol["natural_n"] if stage == "natural" else report["audit"]["accepted"]
        secondary[stage] = {"valid_judgments": len(records), "expected_judgments": count * len(m.METHODS), "methods": {}}
        if len(records) != count * len(m.METHODS):
            secondary_complete = False
            continue
        for method in m.METHODS:
            values = [v for (i, name), v in records.items() if name == method]
            metrics = {"n": len(values)}
            if stage == "natural":
                for key in ("relevance", "information_gain", "non_redundancy", "medical_safety", "overall"):
                    metrics[key] = sum(v[key] for v in values) / len(values)
                for key in ("contains_useful_question", "redundant_question", "unsafe"):
                    metrics[key] = m.wilson([v[key] for v in values])
            else:
                for key in ("missing_target_ask", "known_target_ask"):
                    metrics[key] = m.wilson([v[key] for v in values])
                metrics["paired_success"] = m.wilson([v["missing_target_ask"] and not v["known_target_ask"] for v in values])
            secondary[stage]["methods"][method] = metrics
    report["secondary_judge"] = {"requested_model": m.JUDGES[1], "complete": secondary_complete, **secondary}
    if secondary_complete:
        agreement = {}
        for stage in ("natural", "paired"):
            second = {(r["id"], r["method"]): r["judgment"] for r in judgment_records(stage, m.JUDGES[1]) if r["valid"]}
            keys = ("contains_useful_question", "redundant_question", "unsafe") if stage == "natural" else ("missing_target_ask", "known_target_ask")
            agreement[stage] = {key: m.wilson([a[key] == second[identity][key] for identity, a in judgments[stage].items()]) for key in keys}
        report["judge_exact_binary_agreement"] = agreement
    report["scoring_status"] = "complete_dual_judge" if secondary_complete else "primary_complete_secondary_incomplete"
    m.write_json(m.OUT / "summary.json", report)
    print(json.dumps({"summary": str(m.OUT / "summary.json"), "audit": report["audit"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    generation = sub.add_parser("generate")
    generation.add_argument("--method", choices=m.METHODS, required=True)
    generation.add_argument("--limit", type=int, help="Smoke-test only; manifest remains frozen to this limit")
    judging = sub.add_parser("judge")
    judging.add_argument("--stage", choices=("audit", "natural", "paired"), required=True)
    judging.add_argument("--workers", type=int, default=6)
    judging.add_argument("--judge", choices=m.JUDGES, default=m.JUDGES[0])
    judging.add_argument("--method", choices=m.METHODS, help="Score one completed method while the next generates")
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "prepare":
        p = m.prepare()
        print(json.dumps({k: v for k, v in p.items() if k != "prompts"}, ensure_ascii=False, indent=2))
    elif args.command == "generate":
        generate(args)
    elif args.command == "judge":
        judge(args)
    else:
        summarize()


if __name__ == "__main__":
    main()
