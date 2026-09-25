"""Separate identities for training, evaluation and reproducible source bundles."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import zipfile
from importlib.metadata import version, PackageNotFoundError
from configs.experiment import ROOT
from result.storage import atomic_json, digest


def hash_files(paths):
    h=hashlib.sha256()
    for p in sorted(set(Path(p) for p in paths)):
        h.update(p.relative_to(ROOT).as_posix().encode()+b"\0"+p.read_bytes()+b"\0")
    return h.hexdigest()


def training_files():
    names=['model','observation','learning','preference','rules','training','graph','graph_layers','returns','ppo','future']
    return [ROOT/f'agent/{n}.py' for n in names]+[ROOT/p for p in (
        'configs/experiment.py','configs/config.py','data/online.py','data/benchmark.py','data/generator.py',
        'environment/env.py','environment/public.py','environment/interfaces.py','environment/problem.py',
        'environment/accounting.py','result/storage.py','result/recording.py','result/provenance.py')]


def source_hash(): return hash_files(training_files())


def evaluation_hash():
    return hash_files([*training_files(),ROOT/'result/evaluation.py',ROOT/'agent/offline.py',
                      ROOT/'environment/offline_replay.py',ROOT/'result/offline.py'])


def source_files():
    return sorted(p for folder in ('agent','configs','data','environment','result','scripts','tests','docs')
        for p in (ROOT/folder).glob('**/*') if p.is_file() and p.suffix in ('.py','.yaml','.md')
        and '__pycache__' not in p.parts and not any(x in p.parts for x in ('formal','engineering','instances','raw','runs')))


def full_source_hash(): return hash_files(source_files()+[ROOT/'README.md',ROOT/'requirements.txt'])


def runtime_info():
    versions={}
    for name in ('torch','numpy','scipy','PyYAML','matplotlib','ortools','pytest'):
        try: versions[name]=version(name)
        except PackageNotFoundError: versions[name]=None
    import torch,os
    try:
        git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError):git=None
    return dict(python=platform.python_version(),executable=__import__('sys').executable,os=platform.platform(),
                cpu=platform.processor(),logical_cpus=os.cpu_count(),torch_threads=torch.get_num_threads(),
                cuda=torch.cuda.is_available(),packages=versions,git_commit=git)


def snapshot(run,c):
    run=Path(run);p=run/'source.zip'
    if not p.exists():
        with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
            for f in source_files()+[ROOT/'README.md',ROOT/'requirements.txt']:z.write(f,f.relative_to(ROOT))
            z.writestr('effective_config.json',json.dumps(c,indent=2))
    atomic_json(run/'runtime.json',runtime_info())
    return digest(p)
