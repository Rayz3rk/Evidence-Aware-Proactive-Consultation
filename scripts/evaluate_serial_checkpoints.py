
"""Evaluate the saved epoch checkpoints on the grouped development split."""
from __future__ import annotations
import json, math, sys
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from proactive_consultation.evidence_aware import (EvidenceAwareCollator,EvidenceAwareDataset,EvidenceAwareModel,IGNORE_INDEX,build_examples,build_slot_vocabulary,load_slot_labels,route_for_row,ROUTE_LABELS)
from proactive_consultation.scope3000 import load_views,split_by_pair_id

def metrics(ytrue,ypred,labels):
    conf={str(a):{str(b):0 for b in labels} for a in labels}
    for a,b in zip(ytrue,ypred): conf[str(a)][str(b)]+=1
    fs=[]
    for a in labels:
        tp=conf[str(a)][str(a)]; fp=sum(conf[str(o)][str(a)] for o in labels if o!=a); fn=sum(conf[str(a)][str(o)] for o in labels if o!=a)
        pr=tp/(tp+fp) if tp+fp else 0.; re=tp/(tp+fn) if tp+fn else 0.; f=2*pr*re/(pr+re) if pr+re else 0.; fs.append(f)
    return {'count':len(ytrue),'accuracy':sum(a==b for a,b in zip(ytrue,ypred))/len(ytrue) if ytrue else 0.,'macro_f1':sum(fs)/len(fs) if fs else 0.}

root=ROOT
out=root/'outputs/checkpoints/evidence_aware_scope2000_repaired_1to1_serial_20epoch_saved'
data=root/'data/EviGap-Med/sft_candidates.auto.jsonl'
audit=root/'data/EviGap-Med/paired_contexts.auto.jsonl'
manifest=json.loads((out/'evidence_aware_manifest.json').read_text(encoding='utf-8'))
processor=AutoProcessor.from_pretrained(str(out)); tok=processor.tokenizer
rows=load_views(data,'joint',allowed_quality_statuses={'auto_pass_pending_human'})
pair_slots=load_slot_labels(audit); slots=build_slot_vocabulary(rows,pair_slots); slot_to_id={s:i for i,s in enumerate(slots)}
_,dev_rows=split_by_pair_id(rows,0.1)
examples=build_examples(processor,dev_rows,pair_slots,slot_to_id,max_length=int(manifest.get('max_length',512)))
loader=DataLoader(EvidenceAwareDataset(examples),batch_size=2,shuffle=False,collate_fn=EvidenceAwareCollator(tok.pad_token_id))
model_path=Path(manifest['model_path']); model_path=(out/model_path).resolve() if not model_path.is_absolute() else model_path
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base=AutoModelForImageTextToText.from_pretrained(str(model_path),dtype=(torch.bfloat16 if device.type == 'cuda' else torch.float32),low_cpu_mem_usage=True)
peft_cfg=LoraConfig(task_type=TaskType.CAUSAL_LM,r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['q_proj','k_proj','v_proj','o_proj','in_proj_qkv','out_proj'],bias='none')
backbone=get_peft_model(base,peft_cfg)
model=EvidenceAwareModel(backbone,len(slots),manifest.get('route_loss_weight',.5),manifest.get('slot_loss_weight',.5),route_labels=ROUTE_LABELS,route_class_weights=manifest['route_class_weights'],slot_class_weights=manifest['slot_class_weights'],slot_names=slots)
model.eval(); model.to(device)
results=[]
for step,epoch in [(56,1),(168,3),(1120,20)]:
    state=torch.load(out/f'checkpoint-{step}'/'pytorch_model.bin',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True); del state
    sums=Counter(); n=0; rt=[];rp=[];st=[];sp=[]
    with torch.inference_mode():
      for batch in loader:
        moved={k:v.to(device) for k,v in batch.items()}; o=model(**moved); bs=batch['input_ids'].shape[0]; n+=bs
        for key in ('loss','lm_loss','route_loss','slot_loss'):
          if key in o: sums[key]+=float(o[key])*bs
        rt += batch['route_labels'].tolist(); rp += o['route_logits'].argmax(-1).cpu().tolist()
        valid=batch['slot_labels'].ne(IGNORE_INDEX)
        if valid.any():
          st += batch['slot_labels'][valid].tolist(); sp += o['slot_logits'][valid].argmax(-1).cpu().tolist()
    results.append({'epoch':epoch,'step':step,'eval_loss':sums['loss']/n,'lm_loss':sums['lm_loss']/n,'route_loss':sums['route_loss']/n,'slot_loss':sums['slot_loss']/n,'route_metrics':metrics(rt,rp,sorted(set(rt+rp))),'slot_metrics':metrics(st,sp,list(range(len(slots)))),'dev_examples':n,'slot_supervised_examples':len(st)})
print(json.dumps({'checkpoints':results,'dev_rows':len(dev_rows),'slots':slots},ensure_ascii=False,indent=2))
(root/'outputs/serial_checkpoint_evaluation.json').write_text(json.dumps({'checkpoints':results,'dev_rows':len(dev_rows),'slots':slots},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')



