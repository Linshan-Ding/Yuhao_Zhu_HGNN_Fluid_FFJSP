"""Validate actual files behind completion markers before any successful reuse."""
from pathlib import Path
import json
import torch

from configs.experiment import identity, uses_demo
from result.provenance import source_hash
from result.recording import verify
from result.storage import atomic_json, digest, state_hash


def seal_job(run):
    run = Path(run)
    files = [*run.glob('checkpoint_*.pt'), *run.glob('monitor_*.json')]
    files += [run/name for name in ('budget.json','config.yaml','manifest.json','source.zip',
                                   'raw/manifest.json','log.jsonl','demonstrations.pt',
                                   'initialization_evaluation.json') if (run/name).exists()]
    atomic_json(run/'completion_manifest.json', dict(files={p.relative_to(run).as_posix():digest(p) for p in files}))


def verify_files(root, inventory):
    root = Path(root).resolve()
    for name, sha in inventory.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Artifact path escapes its root: {name}')
        if digest(path) != sha:
            raise ValueError(f'Completed artifact changed: {path}')


def verify_job(run, spec):
    run = Path(run)
    files = json.loads((run/'completion_manifest.json').read_text(encoding='utf-8'))['files']
    required = {f'checkpoint_{spec.budget}.pt','checkpoint_last.pt','checkpoint_best.pt',
                'raw/manifest.json','budget.json','manifest.json','source.zip','config.yaml','log.jsonl'}
    required.update(f'checkpoint_{p}.pt' for p in spec.config['training']['milestones'] if p<=spec.budget)
    if uses_demo(spec.method):
        required.update(('checkpoint_initialization.pt','demonstrations.pt','initialization_evaluation.json'))
    if not required.issubset(files):
        raise ValueError(f'Incomplete job artifacts: {sorted(required-set(files))}')
    verify_files(run, files)
    models=[]
    for name in ('checkpoint_last.pt',f'checkpoint_{spec.budget}.pt'):
        state = torch.load(run/name,weights_only=False,map_location='cpu')
        if (state['source_hash'],state['identity'],state['steps'],state['seed']) != (
                source_hash(),identity(spec.config),spec.budget,spec.seed):
            raise ValueError('Completed job identity mismatch')
        if name=='checkpoint_last.pt' and sum((state['real_steps'],state['scenario_steps'],
                state['demonstrations'].steps,state['lost_upper_bound']))!=spec.budget:
            raise ValueError('Recovery checkpoint budget mismatch')
        models.append(state_hash(state['model']))
    if models[0]!=models[1]:raise ValueError('Final model and recovery checkpoint differ')
    budget=json.loads((run/'budget.json').read_text())
    if budget['charged_upper_bound']!=spec.budget or budget['committed']!=spec.budget:
        raise ValueError('Completed budget is not committed')
    verify(run/'raw')


def verify_assets(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    verify_files(directory,manifest['assets'])


def runtime_key(runtime):
    return {k:runtime[k] for k in ('python','executable','os','cpu','logical_cpus','packages','torch_threads')}


def verify_study(root,data,c,specs):
    from result.statistics import validate
    root=Path(root)
    validate(root,data,c,specs)
    for spec in specs:verify_job(root/'runs'/spec.name,spec)
    for marker in (root/'evaluations').glob('*/complete.json'):
        entry=json.loads(marker.read_text())
        verify(marker.parent/entry['raw_directory'])
        if entry['panel'] and digest(marker.parent/entry['panel'])!=entry['panel_sha256']:
            raise ValueError('Timing panel changed')
    statistics=json.loads((root/'statistics_manifest.json').read_text())
    verify_files(root,statistics['outputs'])
    verify_assets(root/'paper_assets')
    complete=json.loads((root/'complete.json').read_text())
    if digest(root/'paper_assets/manifest.json')!=complete['assets']:
        raise ValueError('Engineering assets manifest changed')
