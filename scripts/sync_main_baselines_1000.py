"""Render complete matched 256-token external-judge results; missing cells stay XXX."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/evaluation/main_baselines_1000_64'
TEX=ROOT/'manuscript/sn-article.tex'
NAMES=['our_epoch3','Qwen3.5-0.8B','Llama-3.2-1B-Instruct','Qwen3-1.7B']
JUDGES=[('deepseek-v4.1-flash','DS-v4.1'),('gpt-5.6-luna','GPT-5.6')]
DIMS=['relevance','information_gain','non_redundancy','medical_safety','proactivity','answerability','conciseness','overall']
def main():
    summaries={}
    for name in NAMES:
        p=OUT/f'{name}.judge_summary.json'
        if p.exists():
            pred=[json.loads(l) for l in (OUT/f'{name}.predictions.jsonl').read_text(encoding='utf8').splitlines() if l.strip()]
            records=[json.loads(l) for l in (OUT/f'{name}.judges.jsonl').read_text(encoding='utf8').splitlines() if l.strip()]
            assert len(pred)==1000 and len(records)==2000
            assert len({(r['example_id'],r['judge']) for r in records})==2000
            summaries[name]=json.loads(p.read_text(encoding='utf8'))['by_source']['all']
    labels={n:('Our method (epoch 3)' if n=='our_epoch3' else n) for n in NAMES}
    text=[r'\begin{table}[htbp]',r'\caption{External-judge comparison on the shared 1,000-context main test.}\label{tab:main_baselines}',r'\centering\footnotesize',r'\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}']
    for i,headers in enumerate([['Relevance','Info. gain','Non-red.','Safety'],['Proactivity','Answerability','Conciseness','Overall']]):
        if i:text.append(r'\vspace{5pt}')
        text.extend([r'\begin{tabularx}{\textwidth}{@{}p{2.9cm} p{1.25cm} *{4}{C}@{}}',r'\toprule','Model & Judge & '+' & '.join(headers)+r' \\',r'\midrule'])
        for judge,short in JUDGES:
            for name in NAMES:
                sub=summaries.get(name,{}).get(judge)
                vals=[f'{sub[d]["mean"]:.2f}' if sub and sub[d]['mean'] is not None else r'\TBD' for d in DIMS[i*4:i*4+4]]
                text.append(labels[name]+' & '+short+' & '+' & '.join(vals)+r' \\')
            if short=='DS-v4.1':text.append(r'\midrule')
        text.extend([r'\bottomrule',r'\end{tabularx}'])
    text.extend([r'\vspace{5pt}',r'\begin{tabularx}{\textwidth}{@{}p{2.9cm} *{5}{C}@{}}',r'\toprule',r'Model & \shortstack{Valid\\question (\%)} & \shortstack{Non-red.\\(\%)} & \shortstack{Unsafe\\(\%)} & \shortstack{Both-accept\\(\%)} & \shortstack{Joint-valid\\($n$)} \\',r'\midrule'])
    for name in NAMES:
        s=summaries.get(name)
        vals=[r'\TBD']*5
        if s:
            vals=[f'{100*sum(s[j][k]["mean"] for j,_ in JUDGES)/2:.1f}' for k in ['valid_question','non_redundant']]
            vals += [f'{100*s[k]["mean"]:.1f}' for k in ['unsafe_output_rate_one_veto','both_accept_rate']]+[str(s['both_judges_valid'])]
        text.append(labels[name]+' & '+' & '.join(vals)+r' \\')
    text.extend([r'\bottomrule',r'\end{tabularx}',r'\tablenote{All models use identical ordered contexts, greedy decoding, a 2,048-token input limit, and the same generation configuration. Judges and scoring rules match Table~\ref{tab:main_rubric}; invalid ratings are excluded, not scored as zero. Pending values are denoted \TBD.}',r'\end{table}'])
    prose=(r'Table~\ref{tab:main_baselines} evaluates our epoch~3 model against the unadapted Qwen3.5-0.8B backbone, Llama-3.2-1B-Instruct, and Qwen3-1.7B under the same main-evaluation protocol. The same-backbone comparison tests the contribution of the complete LoRA+EASC adaptation, while the other baselines provide cross-model context rather than isolate EASC alone. ')
    prose += ('All four evaluations are complete; the table reports judge-specific quality and safety-adjusted behavior without pooling scores into a composite ranking.' if len(summaries)==4 else 'Evaluation is in progress; claims of comparative effectiveness await completed external ratings.')
    rendered='\n'.join(text)+'\n\n'+prose
    s=TEX.read_text(encoding='utf8'); begin='% BEGIN MAIN_BASELINES';end='% END MAIN_BASELINES'
    assert s.count(begin)==s.count(end)==1
    a=s.index(begin)+len(begin);b=s.index(end,a);s=s[:a]+'\n'+rendered+'\n'+s[b:]
    TEX.write_text(s,encoding='utf8')
    print(json.dumps({'completed':list(summaries),'pending':[n for n in NAMES if n not in summaries]}))
if __name__=='__main__':main()
