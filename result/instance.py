"""Single-instance convergence study: every job trains on one fixed small instance and is evaluated on it.

Outputs under the study root: instance.csv (cached greedy evaluations of milestone/final checkpoints and rules),
exact.csv (CP-SAT references, reused from another root when identical), instance_curves.csv (greedy fulfillment and
best training episode against charged interactions), instance_final.csv (final objective per run with gaps to the
reference), instance_summary.csv / instance_effects.csv / instance_per_instance.csv, and paper_assets/.
"""
from pathlib import Path
import json
import numpy as np
from agent.rules import RulePolicy
from configs.experiment import ROOT, ALL_RULES
from data.benchmark import prepare, fixtures
from result.evaluation import cached_case, load_policy, policy_identity
from result.offline import evaluate as exact_reference
from result.provenance import evaluation_hash
from result.statistics import interval, audit
from result.storage import read_csv, write_csv, atomic_json, digest, object_hash

METRICS=('final_greedy_eta','best_episode_eta')


def study_instances(c,data):
    return {i.instance_id:i for i in fixtures(c,data,c['instance_study']['split'])}


def evaluate(root,data,c,specs):
    root=Path(root);index={r['instance_id']:r for r in prepare(c,data)['cases']};instances=study_instances(c,data);rows=[]
    for spec in specs:
        inst=instances[spec.instance]
        for point in sorted({*spec.config['training']['milestones'],spec.budget}):
            path=root/'runs'/spec.name/f'checkpoint_{point}.pt';policy,state=load_policy(path,spec);model_id=policy_identity(state)
            entry=cached_case(root,inst,policy,model_id,c,index[inst.instance_id]['sha256'])
            rows.append(dict(variant=spec.method,run=spec.name,seed=spec.seed,budget=state['steps'],checkpoint_sha256=digest(path),
                             policy_sha256=model_id,evaluation_key=object_hash(entry['identity'])[:24],**entry['row']))
    for rule in ALL_RULES:
        for inst in instances.values():
            entry=cached_case(root,inst,RulePolicy(rule),'rule:'+rule,c,index[inst.instance_id]['sha256'])
            rows.append(dict(variant=rule,run=rule,seed=0,budget=0,checkpoint_sha256='',policy_sha256='rule:'+rule,
                             evaluation_key=object_hash(entry['identity'])[:24],**entry['row']))
    write_csv(root/'instance.csv',rows)
    atomic_json(root/'evaluation_manifest.json',dict(outputs=dict(instance=dict(file='instance.csv',sha256=digest(root/'instance.csv'),rows=len(rows))),
                                                    data=digest(Path(data)/'manifest.json'),evaluation=evaluation_hash()))


def exact(root,data,c):
    s=c['instance_study'];reuse=[ROOT/s['exact_from']] if s.get('exact_from') else []
    exact_reference(root,data,c,splits=[s['split']],reuse=reuse)


def _histories(root,specs):
    for spec in specs:
        yield spec,[json.loads(x) for x in (Path(root)/'runs'/spec.name/'log.jsonl').read_text(encoding='utf-8').splitlines()]


def aggregate(root,data,c,specs):
    root=Path(root);manifest=json.loads((root/'evaluation_manifest.json').read_text())
    if manifest['evaluation']!=evaluation_hash() or manifest['data']!=digest(Path(data)/'manifest.json'):raise ValueError('Evaluation identity changed')
    rows=read_csv(root/'instance.csv')
    if digest(root/'instance.csv')!=manifest['outputs']['instance']['sha256']:raise ValueError('Evaluation table changed')
    exact={r['instance_id']:r for r in read_csv(root/'exact.csv')};curves=[];final=[];last_rows={}
    for spec,history in _histories(root,specs):
        last_rows[spec.name]=history[-1]
        for row in history:
            if row['eta_validation'] is not None:
                curves.append(dict(run=spec.name,method=spec.method,instance_id=spec.instance,seed=spec.seed,steps=row['total_steps'],epoch=row['epoch'],
                    real_steps=row['real_steps'],greedy_eta=row['eta_validation'],best_episode_eta=row['best_episode_eta'],
                    epoch_episode_eta=row['epoch_episode_eta'],episodes=row['episodes']))
    write_csv(root/'instance_curves.csv',curves)
    def gaps(instance_id,eta):
        e=exact[instance_id];reference=float(e['eta']) if e['eta'] else None;upper=float(e['upper']) if e['upper'] else None
        return dict(reference=reference,upper=upper,status=e['status'],gap_to_incumbent=reference-eta if reference is not None else None,
                    gap_to_bound=upper-eta if upper is not None else None)
    learned={(r['run'],int(r['budget'])):r for r in rows if r['seed']!='0'}
    for spec in specs:
        r=learned[spec.name,spec.budget];eta=float(r['eta']);best=last_rows[spec.name]['best_episode_eta']
        final.append(dict(variant=spec.method,run=spec.name,seed=spec.seed,instance_id=spec.instance,final_greedy_eta=eta,
                          best_episode_eta=best,episodes=last_rows[spec.name]['episodes'],wait_fraction=float(r['wait_fraction']),**gaps(spec.instance,eta)))
    for r in rows:
        if r['seed']=='0':
            final.append(dict(variant=r['variant'],run=r['run'],seed=0,instance_id=r['instance_id'],final_greedy_eta=float(r['eta']),best_episode_eta=None,
                              episodes=None,wait_fraction=float(r['wait_fraction']),**gaps(r['instance_id'],float(r['eta']))))
    write_csv(root/'instance_final.csv',final)
    rng=np.random.default_rng(c['experiment']['statistics_seed']);repeats=c['experiment']['bootstrap_repeats']
    methods=sorted({r['variant'] for r in final});ids=sorted({r['instance_id'] for r in final});summary=[];effects=[];per_instance=[]
    def table(method,key):
        selected=[r for r in final if r['variant']==method and r[key] is not None];seeds=sorted({int(r['seed']) for r in selected})
        lookup={(int(r['seed']),r['instance_id']):float(r[key]) for r in selected}
        if set(lookup)!={(s,i) for s in seeds for i in ids}:raise ValueError(f'Missing seed/instance cell: {method} {key}')
        return seeds,np.asarray([[lookup[s,i] for i in ids] for s in seeds])
    for method in methods:
        for key in METRICS:
            if method in ALL_RULES and key!='final_greedy_eta':continue
            seeds,values=table(method,key);means=values.mean(1);lo,hi=interval(means,rng,repeats)
            summary.append(dict(variant=method,metric=key,eta=float(means.mean()),seed_sd=float(means.std(ddof=1)) if len(seeds)>1 else None,
                                ci_low=lo,ci_high=hi,seeds=len(seeds),instances=len(ids)))
        for key in ('gap_to_incumbent','gap_to_bound'):
            vals=[float(r[key]) for r in final if r['variant']==method and r[key] not in (None,'')]
            if vals:summary.append(dict(variant=method,metric=key,eta=float(np.mean(vals)),seed_sd=None,ci_low=None,ci_high=None,seeds=len({r['seed'] for r in final if r['variant']==method}),instances=len(ids)))
        seeds,values=table(method,'final_greedy_eta')
        for j,i in enumerate(ids):
            per_instance.append(dict(variant=method,instance_id=i,eta=float(values[:,j].mean()),seed_sd=float(values[:,j].std(ddof=1)) if len(seeds)>1 else None,seeds=len(seeds)))
    if 'full' in methods:
        fs,fv=table('full','final_greedy_eta')
        for method in methods:
            if method=='full':continue
            ms,mv=table(method,'final_greedy_eta')
            if ms!=[0] and ms!=fs:raise ValueError('Unmatched paired comparison')
            diff=fv-mv;means=diff.mean(1);lo,hi=interval(means,rng,repeats);per=diff.mean(0)
            effects.append(dict(contrast='full minus '+method,difference=float(means.mean()),ci_low=lo,ci_high=hi,positive_seeds=int((means>1e-9).sum()),
                                seeds=len(fs),instances=len(ids),wins=int((per>1e-9).sum()),ties=int((abs(per)<=1e-9).sum()),losses=int((per<-1e-9).sum())))
    write_csv(root/'instance_summary.csv',summary);write_csv(root/'instance_effects.csv',effects);write_csv(root/'instance_per_instance.csv',per_instance)
    trace_audit,evaluation_cost,evaluation_seconds,failed_attempts=audit(root)
    write_csv(root/'trace_audit.csv',trace_audit)
    atomic_json(root/'evaluation_costs.json',dict(unique_complete_evaluations=len(trace_audit),steps=evaluation_cost,
        failed_attempt_steps=sum(r['steps'] for r in failed_attempts),failed_attempts=failed_attempts,seconds=evaluation_seconds,
        training_monitor_steps=sum(int(h['evaluation_steps']) for h in last_rows.values())))
    inputs={k:digest(root/k) for k in ('instance.csv','exact.csv')}
    outputs={p.name:digest(p) for p in root.glob('*.csv') if p.name not in inputs}
    atomic_json(root/'statistics_manifest.json',dict(inputs=inputs,outputs=outputs,seed=c['experiment']['statistics_seed'],
        bootstrap='resample whole training-seed vectors over the fixed instances of the study',
        primary='greedy fulfillment of the final checkpoint on the training instance; best training episode as the optimization view'))


def export(root,data,c,specs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);sm=json.loads((root/'statistics_manifest.json').read_text())
    for name,sha in {**sm['inputs'],**sm['outputs']}.items():
        if digest(root/name)!=sha:raise ValueError('Statistical input/output changed')
    target=root/'paper_assets';target.mkdir(parents=True,exist_ok=True);made=[];micro=c['purpose']=='engineering-smoke'
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,name):
        if micro:fig.suptitle('Engineering smoke only — not performance evidence',fontsize=9)
        fig.tight_layout()
        for ext in ('pdf','svg','png'):
            p=target/f'{name}.{ext}';fig.savefig(p,dpi=300);made.append(p)
        plt.close(fig)
    def esc(x):return str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    def table(name,headers,rows):
        p=target/f'{name}.tex'
        lines=['% Generated from recorded data; '+('SMOKE ONLY' if micro else 'single-instance convergence study'),
               r'\begin{tabular}{'+'l'*len(headers)+'}',r'\toprule',' & '.join(map(esc,headers))+r'\\',r'\midrule']
        lines.extend(' & '.join(esc(v) for v in r)+r'\\' for r in rows)
        lines+=[r'\bottomrule',r'\end{tabular}'];p.write_text('\n'.join(lines),encoding='utf-8');made.append(p)
    for p in root.glob('*.csv'):
        out=target/p.name;__import__('shutil').copy2(p,out);made.append(out)
    curves=read_csv(root/'instance_curves.csv');final=read_csv(root/'instance_final.csv');exact={r['instance_id']:r for r in read_csv(root/'exact.csv')}
    methods=list(c['instance_study']['methods']);ids=sorted({r['instance_id'] for r in final});rules=[r for r in ALL_RULES if any(f['variant']==r for f in final)]
    cols=min(4,len(ids));rows_=int(np.ceil(len(ids)/cols));fig,axes=plt.subplots(rows_,cols,figsize=(4*cols,3*rows_),squeeze=False)
    for ax,instance in zip(axes.flat,ids):
        for method in methods:
            rr=[r for r in curves if r['instance_id']==instance and r['method']==method]
            points=sorted({int(r['steps']) for r in rr})
            if not points:continue
            greedy=[np.mean([float(r['greedy_eta']) for r in rr if int(r['steps'])==p]) for p in points]
            line,=ax.plot(points,greedy,marker='.',ms=3,lw=1,label=method)
            best=[np.mean([float(r['best_episode_eta']) for r in rr if int(r['steps'])==p and r['best_episode_eta']]) if any(r['best_episode_eta'] for r in rr if int(r['steps'])==p) else np.nan for p in points]
            ax.plot(points,best,ls='--',lw=.8,color=line.get_color())
        e=exact.get(instance,{})
        if e.get('eta'):ax.axhline(float(e['eta']),color='black',ls=':',lw=.9,label='CP-SAT incumbent')
        if e.get('upper') and (not e.get('eta') or abs(float(e['upper'])-float(e['eta']))>1e-9):ax.axhline(float(e['upper']),color='gray',ls=':',lw=.9,label='CP-SAT bound')
        best_rule=max((float(f['final_greedy_eta']) for f in final if f['instance_id']==instance and f['variant'] in rules),default=None)
        if best_rule is not None:ax.axhline(best_rule,color='tab:brown',ls='-.',lw=.8,label='best rule')
        ax.set(title=instance,xlabel='Charged training interactions',ylabel='Fulfillment');ax.set_ylim(0,1.02)
    for ax in list(axes.flat)[len(ids):]:ax.axis('off')
    axes.flat[0].legend(fontsize=6,ncol=2);save(fig,'instance_convergence')
    per={(r['variant'],r['instance_id']):r for r in read_csv(root/'instance_per_instance.csv')}
    table('instance_final',['Instance',*methods,'Best rule','CP-SAT incumbent','CP-SAT bound'],
          [[i,*[f"{100*float(per[m,i]['eta']):.1f}"+(f"±{100*float(per[m,i]['seed_sd']):.1f}" if per[m,i]['seed_sd'] else '') if (m,i) in per else '--' for m in methods],
            f"{100*max(float(f['final_greedy_eta']) for f in final if f['instance_id']==i and f['variant'] in rules):.1f}" if rules else '--',
            f"{100*float(exact[i]['eta']):.1f}" if exact[i]['eta'] else '--',f"{100*float(exact[i]['upper']):.1f}" if exact[i]['upper'] else '--'] for i in ids])
    summary={(r['variant'],r['metric']):r for r in read_csv(root/'instance_summary.csv')}
    def cell(m,metric,pct=True):
        r=summary.get((m,metric))
        if r is None:return '--'
        v=100*float(r['eta']) if pct else float(r['eta'])
        return f'{v:.2f}'+(f" [{100*float(r['ci_low']):.2f}, {100*float(r['ci_high']):.2f}]" if r['ci_low'] else '')
    table('instance_summary',['Method','Final greedy (%)','Best episode (%)','Mean gap to incumbent (pp)','Mean gap to bound (pp)'],
          [[m,cell(m,'final_greedy_eta'),cell(m,'best_episode_eta'),cell(m,'gap_to_incumbent'),cell(m,'gap_to_bound')] for m in [*methods,*rules]])
    report=target/'REPORT.md';report.write_text('# '+('工程冒烟：不是正式性能证据' if micro else '单算例收敛研究')+'\n\n'
        '每个作业在同一个固定小规模算例上训练并评测；曲线横坐标为已计费交互，实线为贪心评测履约率，虚线为训练中最佳采样回合。'
        'CP-SAT 可行解与上界、最佳规则为水平参考线。区间只反映训练种子波动。\n',encoding='utf-8');made.append(report)
    atomic_json(target/'manifest.json',dict(purpose=c['purpose'],study='instance',statistics=digest(root/'statistics_manifest.json'),
        data=digest(Path(data)/'manifest.json'),assets={p.name:digest(p) for p in made},source_records='../evaluation_manifest.json'))
    return target
