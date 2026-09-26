"""Frozen formal DAG; stage commands are resumable and require no path arguments."""
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import argparse
import json
import multiprocessing as mp
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
        engineering=hash_files([*sorted((ROOT/'scripts').glob('*.py')),*sorted((ROOT/'tests').glob('*.py')),
            *sorted((ROOT/'result').glob('*.py')),ROOT/'configs/experiment.yaml',ROOT/'requirements.txt'])))


def paths(micro=False,jobs=1):
    if not micro:return ROOT/'result/formal',ROOT/'data/instances/fixed'
    from result.acceptance import runtime_key
    runtime=runtime_info();runtime['torch_threads']=config()['runtime']['threads']
    root=ROOT/'result/engineering'/smoke_identity()[:12]/object_hash(runtime_key(runtime))[:8]/f'jobs_{jobs}'
    return root,ROOT/'data/instances/smoke'


def prepare_study(c,root,data,specs):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);dataset=prepare(c,data)
    value=dict(format='study-freeze-2',training_source=source_hash(),
        jobs=[dict(name=s.name,method=s.method,seed=s.seed,budget=s.budget,group=s.group,
                   config_identity=identity(s.config)) for s in specs],data=dataset['fingerprint'])
    p=root/'freeze.json'
    if p.exists() and json.loads(p.read_text())!=value:
        if list((root/'runs').glob('*/checkpoint_*.pt')):raise ValueError('Frozen training configuration/source/data changed')
    atomic_json(p,value);atomic_json(root/'matrix.json',dict(jobs=[s.record() for s in specs],total=sum(s.budget for s in specs),formal=c['purpose']=='formal'))
    write_csv(root/'run_manifest.csv',[dict(run=s.name,method=s.method,group=s.group,seed=s.seed,budget=s.budget,
        config_identity=identity(s.config),output=f'runs/{s.name}') for s in specs])
    return dataset


_STOP = None


def initialize_worker(stop):
    global _STOP
    _STOP = stop


def worker(spec,root,data):
    from result.engineering import counted
    context=counted(f'worker:{spec.name}') if spec.config['purpose']!='formal' else nullcontext()
    with context:
        return _worker(spec,root,data)


def _worker(spec,root,data):
    from agent.training import train
    from result.recording import verify
    from result.acceptance import seal_job, verify_job
    threads=spec.config['runtime']['threads']
    os.environ['OMP_NUM_THREADS']=str(threads);os.environ['MKL_NUM_THREADS']=str(threads);torch.set_num_threads(threads)
    run=Path(root)/'runs'/spec.name
    with run_lock(run):
        final=run/f'checkpoint_{spec.budget}.pt';status=run/'status.json'
        if status.exists() and json.loads(status.read_text()).get('status')=='complete':
            verify_job(run,spec)
            return dict(run=spec.name,reused=True,status='complete',steps=spec.budget,pid=os.getpid())
        started=time.time()
        atomic_json(status,dict(status='running',pid=os.getpid(),started=started,budget=spec.budget))
        start=time.perf_counter()
        try:
            with (run/'stdout.log').open('a',encoding='utf-8',buffering=1) as f,redirect_stdout(f),redirect_stderr(f):
                result=train(spec.config,run,spec.seed,spec.budget,data,stop_requested=_STOP.is_set if _STOP is not None else None)
            if result['status']!='complete' or result['steps']!=spec.budget or not final.exists():raise ValueError('Job did not reach its exact budget')
            verify(run/'raw');seal_job(run);verify_job(run,spec)
            result.update(run=spec.name,seconds=time.perf_counter()-start,reused=False,
                          pid=os.getpid(),started=started,ended=time.time(),threads=torch.get_num_threads())
            atomic_json(status,result);return result
        except BaseException:
            atomic_json(status,dict(status='failed',error=traceback.format_exc(),seconds=time.perf_counter()-start,
                                    pid=os.getpid(),started=started,ended=time.time()))
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
    selected=list(selected);start=time.perf_counter();results=[];error=None
    try:
        if jobs==1:
            for s in selected:
                print(f'[TRAIN] {s.name} budget={s.budget}',flush=True)
                try:results.append(worker(s,root,data))
                finally:ledger(root,all_specs)
        else:
            ctx=mp.get_context('spawn');stop=ctx.Event()
            pool=ProcessPoolExecutor(max_workers=jobs,mp_context=ctx,initializer=initialize_worker,initargs=(stop,))
            remaining=iter(selected);pending={}
            try:
                for s in list(selected)[:jobs]:pending[pool.submit(worker,s,root,data)]=s
                for _ in range(min(jobs,len(selected))):next(remaining)
                while pending:
                    done,_=wait(pending,return_when=FIRST_COMPLETED)
                    # Observe every completed failure before submitting any replacement job.
                    completed=[f.result() for f in done]
                    for f in done:pending.pop(f)
                    for result in completed:
                        results.append(result);print(f'[DONE] {result["run"]} {len(results)}/{len(selected)}',flush=True)
                    ledger(root,all_specs)
                    for _ in done:
                        s=next(remaining,None)
                        if s is not None:pending[pool.submit(worker,s,root,data)]=s
            except BaseException:
                stop.set()
                for future in pending:future.cancel()
                raise
            finally:
                pool.shutdown(wait=True,cancel_futures=True)
                ledger(root,all_specs)
    except BaseException:
        error=traceback.format_exc()
        raise
    finally:
        p=Path(root)/'execution';p.mkdir(parents=True,exist_ok=True)
        atomic_json(p/f'{time.time_ns()}.json',dict(jobs=jobs,wall_seconds=time.perf_counter()-start,results=results,error=error))
    return results


def storage_preflight(root,c):
    # Scale event data with interactions, and checkpoint/buffer data with their actual units.
    from configs.experiment import uses_demo
    q=ROOT/'result/engineering/complete.json';estimate=0;components={}
    if q.exists() and c['purpose']=='formal':
        proof=json.loads(q.read_text());micro_root=ROOT/proof['root'];small=config(True)
        components['trajectory']=proof['raw_bytes_per_interaction']*c['experiment']['total_budget']
        components['checkpoints_and_buffers']=0
        measurements=json.loads((micro_root/'probes/resources.json').read_text())['methods']
        components['trajectory']=max([proof['raw_bytes_per_interaction'],
            *(r['raw_bytes_per_interaction'] for r in measurements.values())])*c['experiment']['total_budget']
        order_ratio=max(c['data']['orders'])/max(small['data']['orders'])
        for spec in matrix(c):
            measured=measurements[spec.method]
            components['checkpoints_and_buffers']+=measured['recovery_bytes']*c['recording']['keep_recovery_copies']
            components['checkpoints_and_buffers']+=measured['inference_bytes']*(len(spec.config['training']['milestones'])+2)
            if uses_demo(spec.method):components['checkpoints_and_buffers']+=measured['demonstration_bytes']
        components['evaluation_and_reports']=sum(p.stat().st_size for p in (micro_root/'evaluations').rglob('*') if p.is_file())*len(matrix(c))/len(matrix(small))*50/6*order_ratio
        estimate=int(sum(components.values())*c['recording']['preflight_headroom'])
    import shutil
    free=shutil.disk_usage(root).free
    atomic_json(Path(root)/'storage_estimate.json',dict(free_bytes=free,estimated_bytes=estimate,components_before_headroom=components,
        explanation='Measured full-capacity formal buffer/checkpoint sizes plus projected trajectories/evaluation, with configured headroom; not a measured full-run size.'))
    disk_check(root,estimate,c['recording']['minimum_free_gb'])


def memory_preflight(root,jobs):
    from result.resources import memory_info
    proof=json.loads((ROOT/'result/engineering/complete.json').read_text())
    measured=json.loads((ROOT/proof['root']/'probes/resources.json').read_text())
    available=memory_info()['available_bytes'];worker_bytes=measured['worker_with_margin_bytes']
    required=worker_bytes*jobs+4*2**30
    atomic_json(Path(root)/'memory_preflight.json',dict(jobs=jobs,available_bytes=available,required_bytes=required,
        per_worker_bytes=worker_bytes,recommended_jobs=max(0,min(config()['runtime']['jobs'],(available-4*2**30)//worker_bytes))))
    if available<required:raise RuntimeError(f'Insufficient RAM for {jobs} jobs; see {root}/memory_preflight.json and use a lower concurrency')


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
    c=config(micro);specs=matrix(c);root,data=paths(micro,jobs)
    if micro:retire_stale_smoke_data(c,data)
    with run_lock(root):
        prepare_study(c,root,data,specs);ledger(root,specs)
        if stage in ('main','comparators','sensitivity','all'):
            storage_preflight(root,c)
            if not micro:memory_preflight(root,jobs)
            execute([s for s in specs if stage=='all' or s.group==stage],root,data,jobs,specs)
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


def smoke(jobs=1):
    with run_lock(ROOT/'result/engineering/smoke_owner'):
        return _smoke(jobs)


def _smoke(jobs):
    from result.engineering import counted,summarize
    from result.acceptance import verify_study,runtime_key,verify_files
    from scripts.acceptance_probes import run_probes
    torch.set_num_threads(config()['runtime']['threads'])
    signature=smoke_identity();proof=ROOT/'result/engineering/complete.json'
    runtime=runtime_info()
    if proof.exists() and json.loads(proof.read_text())['identity']==signature:
        saved=json.loads(proof.read_text());root=ROOT/saved['root']
        if runtime_key(saved['runtime'])==runtime_key(runtime) and jobs<=saved['validated_jobs']:
            verify_study(root,ROOT/'data/instances/smoke',config(True),matrix(config(True)))
            verify_files(root,saved['evidence'])
            print('[OK] Current engineering acceptance already complete');return saved
    start=time.perf_counter()
    before=summarize()['charged_upper_bound']
    root,_=paths(True,jobs);root.mkdir(parents=True,exist_ok=True)
    result=subprocess.run([sys.executable,'-B','-X','utf8','-m','pytest','tests','-q','--disable-warnings','--maxfail=1',
        f'--junitxml={root / "pytest.xml"}'],cwd=ROOT)
    if result.returncode:raise RuntimeError('Regression tests failed; formal training not started')
    with counted('smoke:parent'):
        run('all',jobs,True)
        # Actual scenario labels must reach the preference optimizer; this is not a performance run.
        from agent.training import train
        c=config(True);probe_budget=c['smoke']['scenario_probe_budget']
        c['training']['milestones']=[probe_budget//2,probe_budget];c['training']['evaluate_every']=probe_budget
        c['data']['due_factor']=list(c['smoke']['probe_due_factor'])  # Engineering-only tight deadlines exercise nonzero preference gradients.
        root,data=paths(True,jobs);probe=root/'scenario_probe';probe_data=probe/'data'
        # Its tighter training distribution has a separate engineering data identity.
        with run_lock(probe):out=train(c,probe,29,c['smoke']['scenario_probe_budget'],probe_data)
        history=[json.loads(x) for x in (probe/'log.jsonl').read_text().splitlines()]
        if not any((r.get('preference_gradient') or 0)>0 and r.get('preference_steps',0)>0 for r in history):raise AssertionError('No nonzero accepted preference learning in smoke')
        raw_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        charged=sum(s.budget for s in matrix(config(True)))+out['steps']
        raw_dirs=[* (root/'runs').glob('*/raw'),probe/'raw']
        trajectory_bytes=sum(p.stat().st_size for folder in raw_dirs for p in folder.rglob('*') if p.is_file())
        recorded_steps=sum(json.loads((folder/'manifest.json').read_text())['steps'] for folder in raw_dirs)
        probes=run_probes(root,data,config(True),matrix(config(True)),jobs)
        probes['pytest.xml']=digest(root/'pytest.xml')
        verify_study(root,data,config(True),matrix(config(True)))
    accounting=summarize()
    saved=dict(identity=signature,root=root.relative_to(ROOT).as_posix(),seconds=time.perf_counter()-start,
        raw_bytes_per_interaction=trajectory_bytes/max(recorded_steps,1),recorded_bytes=raw_bytes,training_smoke_interactions=charged,
        formal_training_started=False,runtime=runtime,validated_jobs=jobs,
        interactions=accounting['charged_upper_bound']-before,engineering_total=accounting['charged_upper_bound'],
        evidence=probes)
    atomic_json(proof,saved)
    print(f'[OK] Engineering acceptance: {proof}',flush=True)
    return saved


def entry(stage):
    runtime=config()['runtime']
    parser=argparse.ArgumentParser(description=f'Fixed-cell scheduling study: {stage}')
    parser.add_argument('jobs',type=int,nargs='?',default=runtime['jobs'],
                        help=f"concurrent CPU jobs; default and upper cap runtime.jobs={runtime['jobs']}");args=parser.parse_args()
    if args.jobs<1:parser.error('concurrency must be a positive integer')
    jobs=min(args.jobs,runtime['jobs'],max(1,os.cpu_count() or 1))
    if jobs!=args.jobs:print(f"[RESOURCE] requested {args.jobs}; using {jobs} CPU jobs, {runtime['threads']} thread(s) each")
    if stage=='smoke':smoke(jobs);return
    if stage in ('main','comparators','sensitivity','all'):smoke(jobs)
    run(stage,jobs)
