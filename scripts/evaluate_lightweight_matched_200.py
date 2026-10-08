"""Matched 200-case / 256-token lightweight comparison, including EASC epoch 3.

One child process at a time releases GPU memory between models. Old 64-token
artifacts are immutable; resumes validate ordered contexts and the full protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/evaluation/local_models_200_256'
OLD = ROOT / 'outputs/evaluation/local_models_200'
RUN = ROOT / 'outputs/checkpoints/evidence_aware_scope2000_repaired_1to1_serial_20epoch_saved'
CHECKPOINT = RUN / 'checkpoint-168/pytorch_model.bin'
MODELS = ['our model', 'Qwen3.5-0.8B', 'Qwen2.5-1.5B-Instruct', 'Qwen3-1.7B',
          'Llama-3.2-1B-Instruct', 'MiniCPM4-0.5B']
KEYS = ['valid_question_rate_rule_proxy', 'non_redundancy_rate_rule_proxy',
        'unsafe_output_rate_rule_proxy', 'average_output_characters', 'average_output_tokens']

def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()] if Path(path).exists() else []

def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def identity(row):
    return {'example_id': row['example_id'], 'context': row['context']}

def prepare():
    manifest = json.loads((OLD / 'dataset_manifest.json').read_text(encoding='utf8'))
    source = ROOT / 'example/data/fused_medical/general/test.jsonl'
    if digest(source) != manifest['sha256']:
        raise RuntimeError('Original diagnostic source hash changed')
    by_id = {r['example_id']: r for r in read(source)}
    rows = [by_id[i] for i in manifest['example_ids']]
    if len(rows) != 200:
        raise RuntimeError('Expected exactly 200 ordered diagnostic cases')
    legacy = read(OLD / 'Qwen3.5-0.8B/predictions.jsonl')
    if [identity(r) for r in rows] != [identity(r) for r in legacy]:
        raise RuntimeError('Frozen context/order mismatch against original predictions')
    protocol = dict(n=200, seed=42, data_sha256=manifest['sha256'],
                    example_ids=manifest['example_ids'], sampling='source-cycled random choice with replacement',
                    max_input_tokens=1024, truncation_side='left', max_new_tokens=256,
                    do_sample=False, batch_size=1, task_directives=False,
                    gpu_process_budget_gib=4, torch_allocator_cap_gib=3.5,
                    max_weight_memory_gib=2.75, cleanup='completed think/analysis blocks and internal controls',
                    evaluator_sha256=digest(__file__))
    target = OUT / 'dataset_manifest.json'
    if target.exists() and json.loads(target.read_text(encoding='utf8')) != protocol:
        raise RuntimeError('New campaign protocol changed; refuse to mix results')
    write(target, protocol)
    return rows, protocol

def child(name):
    rows, protocol = prepare()
    od = OUT / name
    status = od / 'status.json'
    done = read(od / 'predictions.jsonl')
    if len(done) > 200 or [identity(r) for r in done] != [identity(r) for r in rows[:len(done)]]:
        raise RuntimeError('Invalid resume prefix')
    if any(r.get('protocol_sha256') != digest(OUT / 'dataset_manifest.json') for r in done):
        raise RuntimeError('Resume rows have inconsistent protocols')
    if len(done) == 200 and (od / 'summary.json').exists():
        return
    start = time.time()
    write(status, dict(model=name, status='loading', completed=len(done), total=200, pid=os.getpid()))
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        import transformers
        sys.path.insert(0, str(ROOT / 'src'))
        from proactive_consultation.scope3000_evaluation import messages, rule_metrics
        # Avoid importing the modern EASC runtime in the isolated MiniCPM environment.
        import re
        def clean(text):
            text = re.sub(r'<think>.*?</think>|<analysis>.*?</analysis>', '', text, flags=re.I | re.S)
            text = re.sub(r'\[ROUTE=(?:ANSWER|ASK|ANSWER_AND_ASK)\]|\[INQUIRY_SLOT=[^\]]+\]', '', text)
            return text.strip()
        use_gpu = torch.cuda.is_available()
        allocator = 0
        if use_gpu:
            # Leave 512 MiB for non-allocator CUDA overhead and other desktop apps.
            free, total = torch.cuda.mem_get_info()
            allocator = min(int(3.5 * 1024 ** 3), max(0, free - 512 * 1024 ** 2))
            if allocator < 1536 * 1024 ** 2:
                raise RuntimeError('Insufficient free GPU memory; retry after desktop usage falls')
            torch.cuda.set_per_process_memory_fraction(allocator / total, 0)
            weight_budget = min(int(2.75 * 1024 ** 3), allocator - 768 * 1024 ** 2)
            torch.cuda.reset_peak_memory_stats()
        checkpoint_hash = None
        if name == 'our model':
            from proactive_consultation.evidence_aware_runtime import EvidenceAwareRuntime
            runtime = EvidenceAwareRuntime(RUN, device_map='auto' if use_gpu else 'cpu',
                max_input_tokens=1024, max_new_tokens=256,
                max_gpu_memory=weight_budget if use_gpu else None)
            runtime.load()
            state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
            runtime.model.load_state_dict(state, strict=True)
            del state
            runtime.model.eval()
            tok = runtime.tokenizer
            checkpoint_hash = digest(CHECKPOINT)
            model = runtime.model
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            path = ROOT / 'models' / name
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
            tok.truncation_side = 'left'
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            kwargs = dict(trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16 if use_gpu else torch.float32,
                          device_map='auto' if use_gpu else 'cpu', low_cpu_mem_usage=True)
            if use_gpu:
                kwargs.update(max_memory={0: weight_budget, 'cpu': '48GiB'}, offload_folder=str(OUT / '.offload' / name))
            if name == 'Qwen3.5-0.8B':
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
            model.eval()
        version = {'torch': torch.__version__, 'transformers': transformers.__version__, 'python': sys.version, 'executable': sys.executable}
        def update(count, state='generating', **extra):
            write(status, dict(model=name, status=state, completed=count, total=200, pid=os.getpid(),
                elapsed_sec=time.time()-start, gpu_limit='4GiB', allocator_limit_bytes=allocator,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if use_gpu else 0,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if use_gpu else 0, **extra))
        update(len(done))
        with (od / 'predictions.jsonl').open('a', encoding='utf8', newline='\n') as f:
            for i in range(len(done), len(rows)):
                r = rows[i]
                msgs = messages(r)
                if name == 'our model':
                    result = runtime.generate(msgs)
                    text = result.completion
                    extra = dict(route=result.route, slot=result.slot, ask_probability=result.ask_probability)
                    raw = text  # EASC runtime returns cleaned completion.
                else:
                    try:
                        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    except (ValueError, TypeError):
                        prompt = '\n'.join(m['content'] for m in msgs) + '\n'
                    ins = tok(prompt, return_tensors='pt', truncation=True, max_length=1024)
                    dev = model.get_input_embeddings().weight.device
                    ins = {k: v.to(dev) for k, v in ins.items()}
                    with torch.inference_mode():
                        out = model.generate(**ins, max_new_tokens=256, do_sample=False, use_cache=True,
                                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                    raw = tok.decode(out[0][ins['input_ids'].shape[1]:], skip_special_tokens=True)
                    text = clean(raw)
                    extra = {}
                rec = dict(r, prediction=text, raw_output=raw,
                           prediction_tokens=len(tok.encode(text, add_special_tokens=False)),
                           protocol_sha256=digest(OUT / 'dataset_manifest.json'), **extra)
                rec.update(rule_metrics(rec))
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                f.flush()
                update(i+1)
                if (i+1) % 10 == 0 or i+1 == 200:
                    print(f'{name}: {i+1}/200 elapsed={time.time()-start:.1f}s', flush=True)
        vals = read(od / 'predictions.jsonl')
        summary = dict(model=name, status='complete', n=200,
                       metrics={k: sum(float(r[k]) for r in vals)/len(vals) for k in KEYS},
                       epoch=3 if name=='our model' else None,
                       checkpoint=str(CHECKPOINT) if name=='our model' else None,
                       checkpoint_sha256=checkpoint_hash, protocol=protocol, versions=version,
                       elapsed_sec=time.time()-start, gpu_limit='4GiB',
                       peak_reserved_bytes=torch.cuda.max_memory_reserved() if use_gpu else 0,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated() if use_gpu else 0)
        write(od / 'summary.json', summary)
        update(200, 'complete')
    except Exception as e:
        write(status, dict(model=name, status='failed', completed=len(read(od/'predictions.jsonl')),
            total=200, pid=os.getpid(), error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc()))
        raise

def campaign(minicpm_python):
    _, protocol = prepare()
    python = str(ROOT / '.venv/Scripts/python.exe')
    states = {}
    for name in MODELS:
        write(OUT/'campaign_status.json', dict(status='running', model=name, pid=os.getpid(), models=states))
        exe = minicpm_python if name == 'MiniCPM4-0.5B' else python
        if not exe or not Path(exe).exists():
            states[name] = 'blocked: missing isolated MiniCPM interpreter'
            write(OUT/name/'status.json', dict(model=name,status='blocked',completed=0,total=200,error=states[name]))
            continue
        with (OUT/(name.replace(' ','_')+'.log')).open('a',encoding='utf8') as log:
            rc = subprocess.call([exe, '-u', str(Path(__file__).resolve()), '--worker', name],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        states[name] = 'complete' if rc == 0 else f'failed: exit {rc}'
    write(OUT/'campaign_status.json',dict(status='complete' if all(v=='complete' for v in states.values()) else 'incomplete',
                                          pid=os.getpid(), models=states))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', choices=MODELS)
    ap.add_argument('--minicpm-python')
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    if args.prepare_only:
        rows, protocol = prepare()
        print(json.dumps({'n':len(rows),'protocol':protocol}, ensure_ascii=False))
    elif args.worker:
        child(args.worker)
    else:
        campaign(args.minicpm_python)
