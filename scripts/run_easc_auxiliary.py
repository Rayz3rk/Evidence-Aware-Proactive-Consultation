"""Run reduced EASC auxiliary benchmarks for one checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, re, statistics, sys
from pathlib import Path
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from proactive_consultation.evidence_aware_runtime import EvidenceAwareRuntime

def read_jsonl(p):
    p=Path(p)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []
def write_json(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def meddg_rows(path,limit): return read_jsonl(path)[:limit]
def fb_rows(path):
    raw=pq.read_table(path).to_pydict()
    return [{"item_id":f"followupbench_synth_{i:03d}","ehr":raw["EHR"][i],"message":raw["Message"][i],"reference_questions":list(raw["Questions"][i])} for i in range(len(raw["EHR"]))]
def fb_messages(row):
    return [{"role":"system","content":"You are a doctor reading a patient message and medical history. Generate concise English follow-up questions to clarify the patient symptoms and missing medical information. Do not give a diagnosis, treatment advice or answers. Do not repeat information already supplied. Ask one information item per question."},{"role":"user","content":f"Medical history:\n{row['ehr']}\n\nPatient message:\n{row['message']}\n\nWrite exactly 40 follow-up questions as a numbered list. Output only the questions."}]
def clean_questions(text):
    out=[]; seen=set()
    for line in (text or "").splitlines():
        line=re.sub(r"^\s*(?:[-*•]|\d+[.)、:]|[A-Za-z][.)])\s*","",line).strip()
        if not line or not line.endswith(("?","？")) or len(re.findall(r"[?？]",line))!=1: continue
        key=re.sub(r"[^a-z0-9\u4e00-\u9fff]","",line.lower())
        if key and key not in seen: out.append(line); seen.add(key)
    return out
def run_set(runtime, rows, out, kind):
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True); done=read_jsonl(out)
    ids=[r["item_id"] for r in rows]
    if len(done)>len(rows) or [r.get("item_id") for r in done]!=ids[:len(done)]: raise RuntimeError(f"prefix mismatch: {out}")
    with out.open("a",encoding="utf-8",newline="\n") as f:
        for i,row in enumerate(rows[len(done):],start=len(done)):
            messages=row["messages"] if kind=="meddg" else fb_messages(row)
            result=runtime.generate(messages)
            rec=dict(row,raw_output=result.completion,route=result.route,slot=result.slot,output_text=result.completion)
            if kind=="followupbench":
                qs=clean_questions(result.completion); rec.update(generated_questions=qs,number_of_questions=len(qs),rejected_lines=[],questions_over_budget=0)
            f.write(json.dumps(rec,ensure_ascii=False,separators=(",",":"))+"\n"); f.flush()
            if (i+1)%25==0 or i+1==len(rows): print(f"{kind}: {i+1}/{len(rows)}",flush=True)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--label",required=True)
    ap.add_argument("--output-root",type=Path,required=True)
    ap.add_argument("--meddg-limit",type=int,default=500)
    ap.add_argument("--max-gpu-memory",default="8GiB")
    args=ap.parse_args()
    root=args.output_root; root.mkdir(parents=True,exist_ok=True)
    runtime=EvidenceAwareRuntime(args.run_root,device_map="auto",max_input_tokens=2048,max_new_tokens=256,max_gpu_memory=args.max_gpu_memory)
    runtime.load()
    import torch
    state=torch.load(args.checkpoint/"pytorch_model.bin",map_location="cpu",weights_only=True)
    runtime.model.load_state_dict(state,strict=True); del state
    mrows=meddg_rows(ROOT/"example/data/meddg/test.jsonl",args.meddg_limit)
    frows=fb_rows(ROOT/"example/data/followupbench/raw/followupbench_synth.parquet")
    run_set(runtime,mrows,root/f"{args.label}.meddg.predictions.jsonl","meddg")
    runtime.max_new_tokens=1536
    run_set(runtime,frows,root/f"{args.label}.followupbench.predictions.jsonl","followupbench")
    fb=read_jsonl(root/f"{args.label}.followupbench.predictions.jsonl")
    nums=[r.get("number_of_questions",0) for r in fb]
    write_json(root/f"{args.label}.followupbench.rule_scores.json",{"n":len(fb),"question_count":{"mean":statistics.mean(nums) if nums else 0,"median":statistics.median(nums) if nums else 0,"min":min(nums) if nums else 0,"max":max(nums) if nums else 0},"empty_output_rate":sum(n==0 for n in nums)/len(nums) if nums else 0,"rejected_line_count":sum(len(r.get("rejected_lines",[])) for r in fb),"judge_model":None})
    write_json(root/f"{args.label}.manifest.json",{"label":args.label,"checkpoint":str(args.checkpoint.resolve()),"meddg_n":len(mrows),"followupbench_n":len(frows),"max_gpu_memory":args.max_gpu_memory,"protocol":"EASC auxiliary reduced MedDG-500 plus complete FollowupBench-Synth-250; local deterministic generation; no semantic judge"})
    print(json.dumps({"label":args.label,"status":"complete","meddg":len(mrows),"followupbench":len(frows)},ensure_ascii=False))
if __name__=="__main__": main()
