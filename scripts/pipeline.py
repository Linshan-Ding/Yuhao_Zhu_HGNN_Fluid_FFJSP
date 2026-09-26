"""Frozen formal DAG; stage commands are resumable and require no path arguments."""
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import torch
from configs.experiment import ROOT,config,matrix,identity
from data.benchmark import prepare
from result.storage import atomic_json,digest,object_hash,run_lock,disk_check,write_csv
from result.provenance import source_hash,evaluation_hash,hash_files,runtime_info


def smoke_identity():
    return object_hash(dict(training=source_hash(),evaluation=evaluation_hash(),
        engineering=hash_files([ROOT/'scripts/pipeline.py',ROOT/'result/statistics.py',ROOT/'result/export.py',
            ROOT/'configs/experiment.yaml',*sorted((ROOT/'tests').glob('test_*.py'))])))


def paths(micro=False):
    return (ROOT/'result/engineering'/smoke_identity()[:12],ROOT/'data/instances/smoke') if micro else (ROOT/'result/formal',ROOT/'data/instances/fixed')


def prepare_study(c,root,data,specs):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);dataset=prepare(c,data)
    value=dict(training_source=source_hash(),jobs=[s.record() for s in specs],data=dataset['fingerprint'])
    p=root/'freeze.json'
    if p.exists() and json.loads(p.read_text())!=value:
        if list((root/'runs').glob('*/checkpoint_*.pt')):raise ValueError('Frozen training configuration/source/data changed')
    atomic_json(p,value);atomic_json(root/'matrix.json',dict(jobs=[s.record() for s in specs],total=sum(s.budget for s in specs),formal=c['purpose']=='formal'))
    write_csv(root/'run_manifest.csv',[dict(run=s.name,method=s.method,group=s.group,seed=s.seed,budget=s.budget,
        config_identity=identity(s.config),output=f'runs/{s.name}') for s in specs])
    return dataset


def worker(spec,root,data):
    from agent.training import train
    from result.recording import verify
    threads=spec.config['runtime']['threads']
    os.environ['OMP_NUM_THREADS']=str(threads);os.environ['MKL_NUM_THREADS']=str(threads);torch.set_num_threads(threads)
    run=Path(root)/'runs'/spec.name
    with run_lock(run):
        final=run/f'checkpoint_{spec.budget}.pt';status=run/'status.json'
        if status.exists() and json.loads(status.read_text()).get('status')=='complete':
            state=torch.load(final,weights_only=False,map_location='cpu')
            if state['source_hash']!=source_hash() or state['identity']!=identity(spec.config) or state['steps']!=spec.budget or state['seed']!=spec.seed:
                raise ValueError('Completed job identity mismatch')
            verify(run/'raw')
            return dict(run=spec.name,reused=True,status='complete',steps=spec.budget)
        atomic_json(status,dict(status='running',pid=os.getpid(),started=time.time(),budget=spec.budget))
        start=time.perf_counter()
        try:
            with (run/'stdout.log').open('a',encoding='utf-8',buffering=1) as f,redirect_stdout(f),redirect_stderr(f):
                result=train(spec.config,run,spec.seed,spec.budget,data)
            if result['status']!='complete' or result['steps']!=spec.budget or not final.exists():raise ValueError('Job did not reach its exact budget')
            verify(run/'raw');result.update(run=spec.name,seconds=time.perf_counter()-start,reused=False)
            atomic_json(status,result);return result
        except BaseException:
            atomic_json(status,dict(status='failed',error=traceback.format_exc(),seconds=time.perf_counter()-start))
            raise


def ledger(root,specs):
    root=Path(root);rows=[]
    for s in specs:
        p=root/'runs'/s.name/'budget.json';r=json.loads(p.read_text()) if p.exists() else {}
        status=root/'runs'/s.name/'status.json'
        rows.append(dict(run=s.name,budget=s.budget,charged=r.get('charged_upper_bound',0),
            committed=r.get('committed',0),status=json.loads(status.read_text())['status'] if status.exists() else 'not_started'))
    if any(r['charged']>r['budget'] for r in rows):raise ValueError('Job budget exceeded')
    atomic_json(root/'budget_ledger.json',dict(planned=sum(s.budget for s in specs),charged=sum(r['charged'] for r in rows),jobs=rows))


def execute(selected,root,data,jobs,all_specs):
    start=time.perf_counter();results=[]
    if jobs==1:
        for s in selected:
            print(f'[TRAIN] {s.name} budget={s.budget}',flush=True)
            try:results.append(worker(s,root,data))
            finally:ledger(root,all_specs)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pending={pool.submit(worker,s,root,data):s for s in selected}
            try:
                for future in as_completed(pending):
                    result=future.result();results.append(result);ledger(root,all_specs)
                    print(f'[DONE] {result["run"]} {len(results)}/{len(selected)}',flush=True)
            except BaseException:
                for future in pending:future.cancel()
                raise
            finally:ledger(root,all_specs)
    p=Path(root)/'execution';p.mkdir(exist_ok=True)
    atomic_json(p/f'{time.time_ns()}.json',dict(jobs=jobs,wall_seconds=time.perf_counter()-start,results=results))


def storage_preflight(root,c):
    # Scale event data with interactions, and checkpoint/buffer data with their actual units.
    from configs.experiment import uses_demo
    q=ROOT/'result/engineering/complete.json';estimate=0;components={}
    if q.exists() and c['purpose']=='formal':
        proof=json.loads(q.read_text());micro_root=ROOT/proof['root'];small=config(True)
        components['trajectory']=proof['raw_bytes_per_interaction']*c['experiment']['total_budget']
        components['checkpoints_and_buffers']=0
        order_ratio=max(c['data']['orders'])/max(small['data']['orders'])
        for spec in matrix(c):
            base=micro_root/'runs'/f'{spec.method}_s1'
            buffer_ratio=c['classic']['replay_size']/small['classic']['replay_size'] if spec.method in ('dqn','ddqn') else c['training']['rollout_steps']/small['training']['rollout_steps']
            components['checkpoints_and_buffers']+=(base/'checkpoint_last.pt').stat().st_size*order_ratio*buffer_ratio*2
            width_ratio=c['classic']['width']/small['classic']['width'] if spec.method in ('dqn','ddqn','a2c','ppo') else c['network']['width']/small['network']['width']
            components['checkpoints_and_buffers']+=(base/f"checkpoint_{small['experiment']['budget']}.pt").stat().st_size*width_ratio**2*(len(spec.config['training']['milestones'])+2)
            if uses_demo(spec.method):components['checkpoints_and_buffers']+=(base/'demonstrations.pt').stat().st_size*order_ratio*c['demonstration']['steps']/small['demonstration']['steps']
        components['evaluation_and_reports']=sum(p.stat().st_size for p in (micro_root/'evaluations').rglob('*') if p.is_file())*len(matrix(c))/len(matrix(small))*50/6*order_ratio
        estimate=int(sum(components.values())*c['recording']['preflight_headroom'])
    free=disk_check(root,estimate,c['recording']['minimum_free_gb'])
    atomic_json(Path(root)/'storage_estimate.json',dict(free_bytes=free,estimated_bytes=estimate,components_before_headroom=components,
        explanation='Approximate smoke-based trajectory, buffer and checkpoint scaling, with configured headroom; not a measured formal size. Actual usage is checked without dropping data.'))


def retire_stale_smoke_data(c,data):
    """Engineering-only: smoke data generated under an older configuration fingerprint is moved aside, never deleted."""
    from data.benchmark import dataset_identity
    data=Path(data);manifest=data/'manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text(encoding='utf-8'))
        if saved['fingerprint']!=dataset_identity(c):
            stale=data.with_name(f"{data.name}_stale_{saved['fingerprint'][:12]}");data.rename(stale)
            print(f'[DATA] smoke data configuration changed; previous set kept at {stale}',flush=True)


def run(stage,jobs,micro=False):
    c=config(micro);specs=matrix(c);root,data=paths(micro)
    if micro:retire_stale_smoke_data(c,data)
    with run_lock(root):
        prepare_study(c,root,data,specs);ledger(root,specs)
        if stage in ('main','comparators','sensitivity','all'):
            storage_preflight(root,c)
            execute([s for s in specs if stage=='all' or s.group==stage],root,data,1 if micro else jobs,specs)
        if stage in ('evaluate','all'):
            from result.evaluation import evaluate_matrix,latency
            evaluate_matrix(root,data,c,specs);latency(root,c,specs)
        if stage in ('exact','all'):
            from result.offline import evaluate
            evaluate(root,data,c)
        if stage in ('statistics','all'):
            from result.statistics import aggregate
            aggregate(root,data,c,specs)
        if stage in ('export','all'):
            from result.export import export
            export(root,data,c,specs)
        if stage=='all':
            atomic_json(root/'complete.json',dict(purpose=c['purpose'],training=sum(s.budget for s in specs),
                source=source_hash(),assets=digest(root/'paper_assets/manifest.json')))
    print(f'[OK] {stage}: {root}',flush=True)


def smoke():
    from environment.accounting import install
    from result.engineering import InteractionCounter
    signature=smoke_identity();proof=ROOT/'result/engineering/complete.json'
    if proof.exists() and json.loads(proof.read_text())['identity']==signature:
        saved=json.loads(proof.read_text());root=ROOT/saved['root']
        from result.statistics import validate
        validate(root,ROOT/'data/instances/smoke',config(True),matrix(config(True)))
        if digest(root/'paper_assets/manifest.json')!=json.loads((root/'complete.json').read_text())['assets']:raise ValueError('Engineering assets changed')
        print('[OK] Current engineering acceptance already complete');return
    start=time.perf_counter()
    result=subprocess.run([sys.executable,'-B','-X','utf8','-m','pytest','tests','-q','--disable-warnings','--maxfail=1'],cwd=ROOT)
    if result.returncode:raise RuntimeError('Regression tests failed; formal training not started')
    counter=InteractionCounter();install(counter)
    try:
        run('all',1,True)
        # Actual scenario labels must reach the preference optimizer; this is not a performance run.
        from agent.training import train
        c=config(True);probe_budget=c['smoke']['scenario_probe_budget']
        c['training']['milestones']=[probe_budget//2,probe_budget];c['training']['evaluate_every']=probe_budget
        c['data']['due_factor']=list(c['smoke']['probe_due_factor'])  # Engineering-only tight deadlines exercise nonzero preference gradients.
        root,data=paths(True);probe=root/'scenario_probe';probe_data=probe/'data'
        # Its tighter training distribution has a separate engineering data identity.
        with run_lock(probe):out=train(c,probe,29,c['smoke']['scenario_probe_budget'],probe_data)
        history=[json.loads(x) for x in (probe/'log.jsonl').read_text().splitlines()]
        if not any((r.get('preference_gradient') or 0)>0 and r.get('preference_steps',0)>0 for r in history):raise AssertionError('No nonzero accepted preference learning in smoke')
        raw_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        charged=sum(s.budget for s in matrix(config(True)))+out['steps']
        raw_dirs=[* (root/'runs').glob('*/raw'),probe/'raw']
        trajectory_bytes=sum(p.stat().st_size for folder in raw_dirs for p in folder.rglob('*') if p.is_file())
        recorded_steps=sum(json.loads((folder/'manifest.json').read_text())['steps'] for folder in raw_dirs)
        atomic_json(proof,dict(identity=signature,root=str(root.relative_to(ROOT)),seconds=time.perf_counter()-start,
            raw_bytes_per_interaction=trajectory_bytes/max(recorded_steps,1),recorded_bytes=raw_bytes,training_smoke_interactions=charged,
            formal_training_started=False,runtime=runtime_info()))
    finally:install(None);counter.close()
    print(f'[OK] Engineering acceptance: {proof}',flush=True)


def entry(stage):
    runtime=config()['runtime']
    parser=argparse.ArgumentParser(description=f'Fixed-cell scheduling study: {stage}')
    parser.add_argument('jobs',type=int,nargs='?',default=runtime['jobs'],
                        help=f"concurrent CPU jobs; default and upper cap runtime.jobs={runtime['jobs']}");args=parser.parse_args()
    if args.jobs<1:parser.error('concurrency must be a positive integer')
    jobs=min(args.jobs,runtime['jobs'],max(1,os.cpu_count() or 1))
    if jobs!=args.jobs:print(f"[RESOURCE] requested {args.jobs}; using {jobs} CPU jobs, {runtime['threads']} thread(s) each")
    if stage=='smoke':smoke();return
    if stage in ('main','comparators','sensitivity','all'):smoke()
    run(stage,jobs)
