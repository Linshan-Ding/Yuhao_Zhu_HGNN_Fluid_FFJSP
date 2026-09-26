"""Reproducible engineering probes, never formal experiment jobs."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import torch

from configs.experiment import ROOT
from result.acceptance import verify_job, verify_files
from result.storage import atomic_json, digest, state_hash


def run_probes(root,data,c,specs,jobs):
    from scripts.pipeline import prepare_study,execute
    root=Path(root);target=root/'probes';target.mkdir(exist_ok=True)
    serial=target/'serial'
    prepare_study(c,serial,data,specs)
    execute(specs,serial,data,1,specs)
    comparisons=[]
    for spec in specs:
        a=root/'runs'/spec.name;b=serial/'runs'/spec.name
        verify_job(a,spec);verify_job(b,spec)
        for checkpoint in (f'checkpoint_{spec.budget}.pt','checkpoint_last.pt'):
            x=torch.load(a/checkpoint,weights_only=False,map_location='cpu')
            y=torch.load(b/checkpoint,weights_only=False,map_location='cpu')
            if state_hash(x['model'])!=state_hash(y['model']):
                raise AssertionError(f'Serial/parallel model differs: {spec.name}/{checkpoint}')
            for key in ('steps','seed','real_steps','scenario_steps','lost_upper_bound','q_steps','labels','failed_labels'):
                if key in x and x[key]!=y[key]:raise AssertionError(f'Serial/parallel {key} differs')
        comparisons.append(dict(run=spec.name,model_sha256=state_hash(x['model']),steps=x['steps']))
    statuses=[json.loads((root/'runs'/s.name/'status.json').read_text()) for s in specs]
    overlap=max(sum(r['started']<=s['started']<r['ended'] for r in statuses) for s in statuses)
    if jobs>1 and (len({s['pid'] for s in statuses})<2 or overlap<2):
        raise AssertionError('No evidence of overlapping real training processes')
    atomic_json(target/'parallel.json',dict(requested_jobs=jobs,peak_overlap=overlap,
        processes=sorted({r['pid'] for r in statuses}),statuses=statuses,comparisons=comparisons))
    formal=target/'formal';formal.mkdir(exist_ok=True)
    # Fresh subprocesses keep independent peak-memory measurements and release full buffers.
    for method in dict.fromkeys(s.method for s in specs):
        output=formal/method;marker=output/'complete.json'
        if marker.exists():
            saved=json.loads(marker.read_text());verify_files(output,saved['files']);continue
        print(f'[FORMAL-SHAPE] {method}',flush=True)
        output.mkdir(exist_ok=True)
        with (output/'stdout.log').open('w',encoding='utf-8') as log:
            proc=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'scripts/_formal_probe.py'),
                                 method,str(output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError(f'Formal-size probe failed: {output}/stdout.log')
    reports={p.parent.name:json.loads(p.read_text()) for p in formal.glob('*/complete.json')}
    from result.resources import memory_info
    available=memory_info()['available_bytes']
    peak=max(r['peak_memory_bytes'] for r in reports.values())
    # Leave 4 GiB for the OS/parent and apply 25% per-worker margin to the measured peak.
    per_worker=int(peak*1.25)
    recommended=min(jobs,max(0,(available-4*2**30)//max(per_worker,1)))
    resource=dict(available_memory_bytes=available,worker_peak_bytes=peak,worker_with_margin_bytes=per_worker,
                  recommended_jobs=int(recommended),requested_jobs=jobs,methods=reports,
                  interpretation='Measured formal-shape probes with full buffers; not a full-training duration or memory bound.')
    atomic_json(target/'resources.json',resource)
    if recommended<1:raise RuntimeError('Insufficient memory for one formal-size worker with headroom')
    evidence={p.relative_to(root).as_posix():digest(p) for p in target.rglob('*')
              if p.is_file() and p.name not in ('stdout.log','.lock')}
    # Scenario probe also belongs to the acceptance evidence, including raw chunks.
    evidence.update({p.relative_to(root).as_posix():digest(p) for p in (root/'scenario_probe').rglob('*')
                     if p.is_file() and p.name!='.lock'})
    return evidence
