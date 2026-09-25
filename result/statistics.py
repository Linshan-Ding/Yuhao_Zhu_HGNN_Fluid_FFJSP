"""Fixed-grid comparisons; uncertainty is over training seeds, never case replicas."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import wilcoxon
from configs.experiment import ALL_RULES, uses_demo
from data.benchmark import prepare
from result.storage import read_csv, write_csv, atomic_json, digest
from result.recording import records, verify
from result.provenance import evaluation_hash


def validate(root,data,c,specs):
    root=Path(root);dataset=prepare(c,data);m=json.loads((root/'evaluation_manifest.json').read_text())
    if m['data']!=digest(Path(data)/'manifest.json'): raise ValueError('Evaluation data identity changed')
    if m['evaluation']!=evaluation_hash():raise ValueError('Evaluation implementation changed')
    by_name={s.name:s for s in specs};core=[s for s in specs if s.group!='sensitivity'];policy_ids={}
    points={x for s in core for x in s.config['training']['milestones'] if x<s.budget and x>=(64 if c['purpose']=='engineering-smoke' else 100000)}
    required={'main','small','long_stream','large_shop','main_best','initialization','sensitivity_changes','sensitivity_default','sensitivity',*(f'main_{x}' for x in points)}
    if set(m['outputs'])!=required:raise ValueError('Missing evaluation output groups')
    for name,item in m['outputs'].items():
        rows=read_csv(root/item['file'])
        if digest(root/item['file'])!=item['sha256'] or len(rows)!=item['rows']: raise ValueError('Evaluation table changed')
        split=name if name in ('main','small','long_stream','large_shop') else 'validation' if name=='initialization' else 'main'
        ids=set(dataset['splits'][split]);expected=set()
        selected=core
        if name.startswith('sensitivity'):
            selected=[s for s in specs if s.group=='sensitivity'] if name!='sensitivity_default' else []
            if name in ('sensitivity','sensitivity_default'):
                selected += [s for s in core if s.method=='full' and s.seed in c['experiment']['sensitivity_seeds']]
        elif name=='initialization': selected=[s for s in core if uses_demo(s.method)]
        expected.update((s.name,str(s.seed),i) for s in selected for i in ids)
        if name in ('main','small','long_stream','large_shop'):expected.update((r,'0',i) for r in ALL_RULES for i in ids)
        if name=='initialization':expected.update((c['teacher'],'0',i) for i in ids)
        actual=[(r['run'],r['seed'],r['instance_id']) for r in rows]
        if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError(f'Missing/duplicate evaluation cells: {name}')
        for r in rows:
            cache=root/'evaluations'/r['evaluation_key']/'complete.json';entry=json.loads(cache.read_text())
            if entry['row']['instance_id']!=r['instance_id'] or float(r['eta'])!=entry['row']['eta']:raise ValueError('Evaluation row not backed by cache')
            case=next(x for x in dataset['cases'] if x['instance_id']==r['instance_id'])
            if entry['identity']['evaluation']!=evaluation_hash() or entry['identity']['instance']!=case['sha256']:raise ValueError('Evaluation cache identity changed')
            if digest(cache.parent/entry['raw_manifest'])!=entry['raw_sha256']:raise ValueError('Raw manifest changed')
            if r['run'] in by_name:
                filename='checkpoint_best.pt' if name=='main_best' else 'checkpoint_initialization.pt' if name=='initialization' else f"checkpoint_{r['budget']}.pt"
                if digest(root/'runs'/r['run']/filename)!=r['checkpoint_sha256']:raise ValueError('Checkpoint changed')
                from result.evaluation import load_policy,policy_identity
                key=(r['run'],filename)
                if key not in policy_ids:
                    _,state=load_policy(root/'runs'/r['run']/filename,by_name[r['run']]);policy_ids[key]=policy_identity(state)
                if entry['identity']['model']!=r['policy_sha256'] or r['policy_sha256']!=policy_ids[key]:raise ValueError('Cached model identity mismatch')
                expected_budget=c['experiment']['sensitivity_budget'] if name=='sensitivity_default' or name=='sensitivity' and by_name[r['run']].group!='sensitivity' else int(name[5:]) if name.startswith('main_') and name!='main_best' else by_name[r['run']].budget
                if name not in ('main_best','initialization') and int(r['budget'])!=expected_budget:raise ValueError('Evaluation budget mismatch')
    exact=json.loads((root/'exact_manifest.json').read_text())
    if set(exact['cases'])!={i+'.json' for i in dataset['splits']['small']}:raise ValueError('Missing offline reference cases')
    for filename,sha in exact['cases'].items():
        if digest(root/'exact'/filename)!=sha:raise ValueError('Offline result changed')
    for filename,sha in exact['outputs'].items():
        if digest(root/filename)!=sha:raise ValueError('Offline table changed')
    timing=json.loads((root/'latency_manifest.json').read_text())
    if timing['identity']['evaluation']!=evaluation_hash() or timing['sha256']!=digest(root/'latency.csv'):raise ValueError('Latency cache changed')
    return m


def grouped(rows,group):
    return [r for r in rows if group=='all' or group=='frequent' and float(r['iota'])>=1 or
            group=='infrequent' and float(r['iota'])<1 or group=='overload' and float(r['rho'])>=1 or
            group=='underload' and float(r['rho'])<1]


def seed_values(rows,method):
    selected=[r for r in rows if r['variant']==method];seeds=sorted({int(r['seed']) for r in selected})
    ids=sorted({r['instance_id'] for r in selected});lookup={(int(r['seed']),r['instance_id']):float(r['eta']) for r in selected}
    return seeds,ids,np.asarray([[lookup[s,i] for i in ids] for s in seeds])


def interval(values,rng,repeats):
    if len(values)<2:return None,None
    samples=values[rng.integers(len(values),size=(repeats,len(values)))].mean(1)
    return float(np.quantile(samples,.025)),float(np.quantile(samples,.975))


def aggregate(root,data,c,specs):
    root=Path(root);manifest=validate(root,data,c,specs);rng=np.random.default_rng(c['experiment']['statistics_seed'])
    summaries=[];effects=[];case_rows=[];seed_rows=[];tests=[]
    for split in ('main','small','long_stream','large_shop','sensitivity','main_best','initialization'):
        rows=read_csv(root/f'{split}.csv');methods=sorted({r['variant'] for r in rows})
        for method in methods:
            seeds,ids,values=seed_values(rows,method)
            for j,inst in enumerate(ids):
                original=next(r for r in rows if r['variant']==method and r['instance_id']==inst)
                case_rows.append(dict(split=split,variant=method,instance_id=inst,eta=float(values[:,j].mean()),
                    seed_sd=float(values[:,j].std(ddof=1)) if len(seeds)>1 else None,seeds=len(seeds),
                    **{k:original[k] for k in ('orders','products','stages','machines','iota','rho','due_factor')}))
        for group in ('all','frequent','infrequent','overload','underload'):
            sub=grouped(rows,group)
            if not sub:continue
            for method in methods:
                seeds,ids,values=seed_values(sub,method);means=values.mean(1);lo,hi=interval(means,rng,c['experiment']['bootstrap_repeats'])
                summaries.append(dict(split=split,group=group,variant=method,eta=float(means.mean()),
                    seed_sd=float(means.std(ddof=1)) if len(seeds)>1 else None,ci_low=lo,ci_high=hi,seeds=len(seeds),instances=len(ids),
                    uncertainty='training-seed variation conditional on fixed parameter grid'))
                seed_rows.extend(dict(split=split,group=group,variant=method,seed=s,eta=float(v),instances=len(ids)) for s,v in zip(seeds,means))
            if 'full' not in methods:continue
            ss,ids,av=seed_values(sub,'full');pairs=[]
            for method in methods:
                if method=='full':continue
                bs,bids,bv=seed_values(sub,method)
                if ids!=bids or (bs!=[0] and bs!=ss): raise ValueError('Unmatched paired comparison')
                diff=av-bv;means=diff.mean(1);lo,hi=interval(means,rng,c['experiment']['bootstrap_repeats']);per_case=diff.mean(0)
                effects.append(dict(split=split,group=group,contrast='full minus '+method,difference=float(means.mean()),ci_low=lo,ci_high=hi,
                    positive_seeds=int((means>1e-9).sum()),seeds=len(ss),instances=len(ids),wins=int((per_case>1e-9).sum()),
                    ties=int((abs(per_case)<=1e-9).sum()),losses=int((per_case< -1e-9).sum())))
                p=1. if np.allclose(means,0) else float(wilcoxon(means,zero_method='wilcox',method='auto').pvalue)
                pairs.append(dict(split=split,group=group,comparison='full minus '+method,p=p,seeds=len(ss),unit='training seed; fixed grid',
                                  test='two-sided paired Wilcoxon; small-sample conditional inference'))
            last=0.
            for k,r in enumerate(sorted(pairs,key=lambda r:r['p'])):last=max(last,min(1.,r['p']*(len(pairs)-k)));r['p_holm']=last
            tests.extend(pairs)
    for name,rows in [('summary',summaries),('effects',effects),('per_case',case_rows),('per_seed',seed_rows),('tests',tests)]:write_csv(root/f'{name}.csv',rows)
    curves=[];costs=[];preference=[]
    for spec in specs:
        run=root/'runs'/spec.name;history=[json.loads(x) for x in (run/'log.jsonl').read_text().splitlines()]
        for row in history:
            if row['eta_validation'] is not None:curves.append(dict(run=spec.name,method=spec.method,seed=spec.seed,steps=row['total_steps'],eta=row['eta_validation']))
        last=history[-1];timing={'on_policy_seconds':0.,'preference_seconds':0.,'q_seconds':0.}
        for update in records(run/'raw','updates'):
            for key in timing:
                timing[key]+=float(update.get(key,update.get('seconds',0.) if key=='q_seconds' and update['kind']=='q' else 0.))
        costs.append(dict(run=spec.name,method=spec.method,seed=spec.seed,group=spec.group,**{k:last[k] for k in
              ('total_steps','real_steps','scenario_steps','demonstration_steps','lost_upper_bound','evaluation_steps','elapsed_seconds','scenario_seconds','recording_seconds')},**timing))
        for label in records(run/'raw','preferences'):
            preference.append(dict(run=spec.name,query_id=label['query_id'],complete=label['complete'],steps=label['steps'],
                mean=label['mean'],standard_error=label['standard_error'],reliability=label['reliability'],probability=label['probability'],source=label['source']))
    write_csv(root/'learning_curves.csv',curves);write_csv(root/'costs.csv',costs)
    write_csv(root/'preference_diagnostics.csv',preference,['run','query_id','complete','steps','mean','standard_error','reliability','probability','source'])
    write_csv(root/'behavior.csv',[{k:v for k,v in r.items() if k in ('variant','seed','instance_id','wait_fraction','held_share') or any(k.startswith(p+'_') for p in ('startup','arrivals','drain'))} for r in read_csv(root/'main.csv')])
    exact={r['instance_id']:r for r in read_csv(root/'exact.csv')};gaps=[]
    for r in read_csv(root/'small.csv'):
        e=exact[r['instance_id']];reference=float(e['eta']) if e['eta'] else None
        gaps.append(dict(variant=r['variant'],seed=r['seed'],instance_id=r['instance_id'],eta=r['eta'],reference=reference,
            upper=e['upper'],status=e['status'],incumbent_difference=reference-float(r['eta']) if reference is not None else None))
    write_csv(root/'offline_gaps.csv',gaps)
    trace_audit=[];evaluation_cost=0;evaluation_seconds=0.;complete_raw=set()
    for p in sorted((root/'evaluations').glob('*/complete.json')):
        entry=json.loads(p.read_text());raw=p.parent/entry['raw_directory'];verify(raw);row=entry['row'];rewards=0.;completed=0
        held=0.;waits=0;starts={};finishes=set();busy=np.zeros(int(row['machines']));last_end=np.zeros_like(busy)
        dynamic={k:np.array(v) for k,v in next(records(raw,'episodes'))['initial'].items()}
        for d in records(raw,'decisions'):
            rewards+=d['reward'];waits+=int(d['action'].get('wait',False))
            for key,change in d['delta'].items():dynamic[key][change['index']]=change['value']
        for e in records(raw,'events'):
            completed+=int(e['kind']=='order_resolved' and e.get('outcome')==1)
            if e['kind']=='wait':held+=e['held_machine_time']
            if e['kind']=='operation_start':
                key=(e['order'],e['stage']);machine=e['machine']
                if key in starts or e['time']<last_end[machine]-1e-7:raise ValueError('Overlapping or duplicated raw operations')
                starts[key]=e;last_end[machine]=e['time']+e['duration'];busy[machine]+=e['duration']
            if e['kind']=='operation_finish':
                key=(e['order'],e['stage']);start=starts[key]
                if abs(start['time']+start['duration']-e['time'])>1e-7:raise ValueError('Raw nonpreemptive duration mismatch')
                finishes.add(key)
        if abs(rewards-row['eta'])>1e-9 or completed!=row['completed']:raise ValueError(f'Raw trace disagrees with fulfillment: {p}')
        if abs(waits/max(row['steps'],1)-row['wait_fraction'])>1e-9 or abs(held/max(row['machines']*row['simulation_end'],1e-9)-row['held_share'])>1e-9:raise ValueError('Raw waiting metrics mismatch')
        if set(starts)!=finishes or not np.allclose(busy,dynamic['machine_busy_time']) or int(np.sum(dynamic['order_outcome']==1))!=completed:raise ValueError('Raw schedule/state mismatch')
        trace_audit.append(dict(evaluation_key=p.parent.name,reward_sum=rewards,eta=row['eta'],completed=completed,
                               wait_actions=waits,held_machine_time=held,operations=len(starts),passed=True))
        evaluation_cost+=row['steps'];evaluation_seconds+=row['wall_seconds']
        complete_raw.add(raw.resolve())
    failed_attempts=[]
    for manifest_path in sorted((root/'evaluations').glob('*/attempt_*/raw/manifest.json')):
        if manifest_path.parent.resolve() in complete_raw:continue
        state=verify(manifest_path.parent)
        failed_attempts.append(dict(raw=manifest_path.parent.relative_to(root).as_posix(),steps=state['steps']))
    write_csv(root/'trace_audit.csv',trace_audit)
    atomic_json(root/'evaluation_costs.json',dict(unique_complete_evaluations=len(trace_audit),steps=evaluation_cost,
        failed_attempt_steps=sum(r['steps'] for r in failed_attempts),failed_attempts=failed_attempts,seconds=evaluation_seconds,
        training_monitor_steps=sum(int(r['evaluation_steps']) for r in costs),offline_replay_steps=sum(int(r['replay_steps']) for r in exact.values())))
    inputs={k:digest(root/k) for k in [*(v['file'] for v in manifest['outputs'].values()),'exact.csv','latency.csv']}
    outputs={p.name:digest(p) for p in root.glob('*.csv') if p.name not in inputs}
    atomic_json(root/'statistics_manifest.json',dict(inputs=inputs,outputs=outputs,seed=c['experiment']['statistics_seed'],
        bootstrap='resample whole training-seed vectors; no resampling of fixed cases',comparison='conditional component effects; no factorial interaction'))
