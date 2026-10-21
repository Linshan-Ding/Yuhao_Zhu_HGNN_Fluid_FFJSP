"""Pre-registered pilot verdict, computed only from persisted statistics and training logs (docs/preregistration-v2.md)."""
from pathlib import Path
import json
import numpy as np
from result.storage import read_csv, atomic_json
from result.evaluation import PRIMARY

HYPOTHESES=(('H1 competition correction','no_graph'),('H2 scenario learning','no_scenario'),('H3 demonstration initialization','no_demo'))


def verdict(root,c,specs):
    root=Path(root);t=c['pilot']['thresholds'];v=c['pilot']['validity']
    effects={r['contrast']:r for r in read_csv(root/'effects.csv') if r['split']==PRIMARY and r['group']=='all'}
    summary={r['variant']:r for r in read_csv(root/'summary.csv') if r['split']==PRIMARY and r['group']=='all'}
    hypotheses=[]
    for name,ablation in HYPOTHESES:
        r=effects.get('full minus '+ablation)
        if r is None:hypotheses.append(dict(hypothesis=name,contrast='full minus '+ablation,status='not_tested',passed=None));continue
        gain=100*float(r['difference']);positive=int(r['positive_seeds']);seeds=int(r['seeds'])
        hypotheses.append(dict(hypothesis=name,contrast=r['contrast'],gain_pp=gain,
            ci_pp=[100*float(r['ci_low']) if r['ci_low'] else None,100*float(r['ci_high']) if r['ci_high'] else None],
            positive_seeds=f'{positive}/{seeds}',cells=f"{r['wins']}/{r['ties']}/{r['losses']}",
            passed=bool(gain>=t['mechanism_gain_pp'] and positive>=min(int(t['positive_seeds']),seeds))))
    comparators={m:100*float(r['eta']) for m,r in summary.items() if m!='full'}
    full=100*float(summary['full']['eta']) if 'full' in summary else None
    strongest=max(comparators,key=comparators.get) if comparators else None
    hypotheses.append(dict(hypothesis='H4 parity with the strongest comparator',full_pp=full,strongest=strongest,
        strongest_pp=comparators.get(strongest),margin_pp=t['baseline_margin_pp'],
        passed=None if full is None or strongest is None else bool(full>=comparators[strongest]-t['baseline_margin_pp'])))
    checks=[]
    for spec in specs:
        if spec.method!='full':continue
        history=[json.loads(x) for x in (root/'runs'/spec.name/'log.jsonl').read_text(encoding='utf-8').splitlines()]
        last=history[-1];tail=[r['entropy'] for r in history[-10:] if r.get('entropy') is not None]
        entropy=float(np.mean(tail)) if tail else None
        checks.append(dict(run=spec.name,labels=last['labels'],usable_labels=last['usable_labels'],contested_checks=last.get('contested_checks'),
            final_entropy=entropy,usable_ok=bool(last['usable_labels']>=v['minimum_usable_labels']),
            entropy_ok=None if entropy is None else bool(entropy>=v['minimum_entropy'])))
    rows=[r for r in read_csv(root/f'{PRIMARY}.csv') if r['variant']=='full']
    wait=float(np.mean([float(r['wait_fraction']) for r in rows])) if rows else None
    per_seed=[float(r['eta']) for r in read_csv(root/'per_seed.csv') if r['split']==PRIMARY and r['group']=='all' and r['variant']=='full']
    collapse=bool(per_seed and min(per_seed)<float(np.median(per_seed))-.05)
    validity=dict(runs=checks,mean_wait_fraction=wait,wait_ok=None if wait is None else bool(wait<=v['maximum_wait_fraction']),
                  seed_collapse=collapse,per_seed_pp=[100*x for x in per_seed])
    tested=[h for h in hypotheses if h.get('passed') is not None]
    decision='proceed_to_formal' if tested and all(h['passed'] for h in tested) else 'iterate_or_demote'
    out=dict(decision=decision,hypotheses=hypotheses,validity=validity,thresholds=t,validity_thresholds=v,
             rule='all pre-registered hypotheses must pass; a failed mechanism gets at most one redesign and second pilot, then becomes an ablation')
    atomic_json(root/'pilot_verdict.json',out);print(f'[PILOT] decision={decision}',flush=True)
    return out
