"""Fill the manuscript's matched diagnostic from complete, verified summaries only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/evaluation/local_models_200_256'
TEX = ROOT / 'manuscript/sn-article.tex'
NAMES = ['our model', 'Qwen3.5-0.8B', 'Qwen2.5-1.5B-Instruct', 'Qwen3-1.7B',
         'Llama-3.2-1B-Instruct', 'MiniCPM4-0.5B']
KEYS = ['valid_question_rate_rule_proxy', 'non_redundancy_rate_rule_proxy',
        'unsafe_output_rate_rule_proxy', 'average_output_characters', 'average_output_tokens']

def main():
    protocol = json.loads((OUT/'dataset_manifest.json').read_text(encoding='utf8'))
    phash = hashlib.sha256((OUT/'dataset_manifest.json').read_bytes()).hexdigest()
    completed = {}
    for name in NAMES:
        if name == 'MiniCPM4-0.5B':
            continue
        path = OUT/name/'summary.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding='utf8'))
        rows = [json.loads(l) for l in (OUT/name/'predictions.jsonl').read_text(encoding='utf8').splitlines() if l.strip()]
        assert summary['status'] == 'complete' and summary['n'] == len(rows) == 200
        assert summary['protocol'] == protocol
        assert [r['example_id'] for r in rows] == protocol['example_ids']
        assert all(r['protocol_sha256'] == phash for r in rows)
        assert all(abs(summary['metrics'][k] - sum(float(r[k]) for r in rows)/200) < 1e-10 for k in KEYS)
        if name == 'our model':
            assert summary['epoch'] == 3 and Path(summary['checkpoint']).parent.name == 'checkpoint-168'
        completed[name] = summary['metrics']
    legacy = json.loads((ROOT/'outputs/evaluation/local_models_200/MiniCPM4-0.5B/summary.json').read_text(encoding='utf8'))
    assert legacy['status'] == 'complete' and legacy['n'] == 200
    completed['MiniCPM4-0.5B'] = legacy['metrics']
    table_rows = []
    for i, name in enumerate(NAMES):
        if i == 1:
            table_rows.append(r'\midrule')
        m = completed.get(name)
        vals = ([f'{100*m[k]:.1f}' for k in KEYS[:3]] + [f'{m[k]:.2f}' for k in KEYS[3:]]) if m else [r'\TBD']*5
        table_rows.append((name + (r'$^{\dagger}$' if name == 'MiniCPM4-0.5B' else '')) + ' & ' + ' & '.join(vals) + r' \\')
    common = (r'These formatting, repetition, and lexical-risk proxies are not numerically interchangeable with the external-judge ratings in Table~\ref{tab:main_rubric}.')
    if 'our model' in completed:
        m = completed['our model']
        analysis = (r'As shown in Table~\ref{tab:local_models}, our model at epoch~3 achieves a '
                    + f"{100*m[KEYS[0]]:.1f}" + r'\% rule-retained valid-question rate, a '
                    + f"{100*m[KEYS[1]]:.1f}" + r'\% non-redundancy rate, and a '
                    + f"{100*m[KEYS[2]]:.1f}" + r'\% unsafe-output flag rate on the shared diagnostic set. ')
        baselines = {n:v for n,v in completed.items() if n not in ('our model', 'MiniCPM4-0.5B')}
        if len(baselines) == 4:
            best = max(v[KEYS[0]] for v in baselines.values())
            low = min(v[KEYS[0]] for v in baselines.values())
            analysis += f'The four matched baselines have valid-question rates of {100*low:.1f}' + r'\% to ' + f'{100*best:.1f}' + r'\%. '
            if m[KEYS[0]] > best:
                analysis += (f'Our model exceeds the strongest baseline on this rule by {100*(m[KEYS[0]]-best):.1f} percentage points, '
                             'supporting the benefit of explicit inquiry-oriented adaptation for producing a complete single-question output. ')
            else:
                analysis += 'The rule-based ordering is reported separately from semantic inquiry quality, for which the external judges provide the relevant evidence. '
        else:
            analysis += 'The remaining matched baseline evaluations are pending, so a complete cross-model ranking is not yet available. '
        analysis += (f'The mean cleaned output contains {m[KEYS[3]]:.2f} characters and {m[KEYS[4]]:.2f} tokens; '
                     'output length is descriptive rather than a quality score. ' + common)
    else:
        analysis = (r'Table~\ref{tab:local_models} provides a matched comparison of output behavior: valid-question rate measures retention under the single-question formatting rule, non-redundancy detects verbatim reuse, and the unsafe-output rate reflects predefined lexical flags. Output length is reported to distinguish concise inquiry generation from longer responses. The matched evaluation is in progress; comparative conclusions await completed rows. ' + common)
    text = TEX.read_text(encoding='utf8')
    for kind, content in [('ROWS', '\n'.join(table_rows)), ('ANALYSIS', analysis)]:
        begin = '% BEGIN MATCHED LIGHTWEIGHT ' + kind
        end = '% END MATCHED LIGHTWEIGHT ' + kind
        assert text.count(begin) == text.count(end) == 1
        a = text.index(begin) + len(begin)
        b = text.index(end, a)
        text = text[:a] + '\n' + content + '\n' + text[b:]
    temp = TEX.with_suffix('.tex.tmp')
    temp.write_text(text,encoding='utf8')
    temp.replace(TEX)
    print(json.dumps({'completed':list(completed),'pending':[n for n in NAMES if n not in completed]},ensure_ascii=False))

if __name__ == '__main__':
    main()
