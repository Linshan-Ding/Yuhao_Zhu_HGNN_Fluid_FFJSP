"""Fixed-input regression snapshot for refactors that must not change any number.

    python scripts/_regression_snapshot.py --save  PATH   # before the refactor
    python scripts/_regression_snapshot.py --check PATH   # after; exits non-zero on any difference

What is frozen (all with fixed seeds, on the CPU, one thread):
  * observation arrays of two fixed environments (a two-order toy and an online-sampled instance);
  * one online-sampled training instance and the micro fixed benchmark (instance arrays, not file hashes);
  * forward outputs (logits, value) of the formal-size policy of all methods on those observations;
  * final weights and numeric log fields of a 32-interaction micro training run of every method.
Hashes, timings and identities are excluded on purpose: they are allowed to change.
"""
import argparse
import dataclasses
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402

from configs.experiment import METHODS, config, environment_config  # noqa: E402
from data.benchmark import prepare, fixtures  # noqa: E402
from data.generator import Instance  # noqa: E402
from data.online import sample  # noqa: E402
from environment.public import SchedulingEnv  # noqa: E402
from agent.observation import observe  # noqa: E402
from agent.model import Policy  # noqa: E402

VOLATILE = ('second', 'time', 'hash', 'sha', 'source', 'wall', 'elapsed', 'identity', 'bytes', 'commit')
# Log fields removed (always 0) or renamed by the 2026-09-26 tidy-up; applied to the 'before' snapshot on --check.
DROPPED = ('scenario_loss', 'labelled_states', 'label_fraction')
RENAMED = {'teacher_version': 'frozen_version'}


def arrays(obj, prefix=''):
    """Flatten dataclasses / dicts / tuples into {name: numpy array or scalar}."""
    out = {}
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            out.update(arrays(getattr(obj, f.name), f'{prefix}{f.name}.'))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(arrays(v, f'{prefix}{k}.'))
    elif isinstance(obj, (list, tuple)) and obj and not isinstance(obj[0], (int, float, str)):
        for i, v in enumerate(obj):
            out.update(arrays(v, f'{prefix}{i}.'))
    elif isinstance(obj, np.ndarray):
        out[prefix.rstrip('.')] = obj.copy()
    elif isinstance(obj, (int, float, bool, str, np.generic)):
        out[prefix.rstrip('.')] = obj
    elif isinstance(obj, (list, tuple)):
        out[prefix.rstrip('.')] = np.asarray(obj)
    return out


def tiny_env(c, alternative=True):
    i = Instance('tiny', 'test', 2, 1, (2,), np.array([[5, 5], [6, 6 if alternative else 0]], np.float32),
                 np.array([0, 1]), np.array([0., 0.]), np.array([100., 80.]),
                 {'DDT': 10., 'iota': 1., 'rho_sys': 1., 'due_factor': 2.})
    return SchedulingEnv(i, environment_config(c))


def snapshot():
    torch.set_num_threads(1)
    snap = {}
    formal = config()
    envs = {'tiny': tiny_env(formal), 'tiny_noalt': tiny_env(formal, False),
            'online71': SchedulingEnv(sample(formal, np.random.default_rng(71)), environment_config(formal))}
    obs = {k: observe(e) for k, e in envs.items()}
    for k, o in obs.items():
        snap[f'obs/{k}'] = arrays(o)
    snap['instance/online71'] = arrays(sample(formal, np.random.default_rng(71)))
    with tempfile.TemporaryDirectory() as tmp:
        micro = config(True)
        m = prepare(micro, Path(tmp) / 'micro')
        snap['bench/micro/splits'] = {k: list(v) for k, v in m['splits'].items()}
        for split in ('validation', 'main', 'small', 'long_stream', 'large_shop'):
            for inst in fixtures(micro, Path(tmp) / 'micro', split):
                snap[f'bench/micro/{inst.instance_id}'] = arrays(inst)
    for method in METHODS:
        c = deepcopy(formal); c['method'] = method
        torch.manual_seed(0); p = Policy(c); p.eval()
        for k, o in obs.items():
            with torch.no_grad():
                out = p(p.batch([o, o]))
            snap[f'forward/{method}/{k}/logits'] = out.logits.numpy().copy()
            if out.value is not None:
                snap[f'forward/{method}/{k}/value'] = out.value.numpy().copy()
    from agent.training import train
    for method in METHODS:
        c = config(True); c['method'] = method
        c['demonstration'].update(steps=8, epochs=1)
        c['training'].update(milestones=[16, 32], evaluate_every=32, rollout_steps=8)
        c['classic'].update(start=8, batch=4, train_every=4)
        with tempfile.TemporaryDirectory() as tmp:
            train(c, Path(tmp) / 'run', 3, 32, Path(tmp) / 'data')
            ck = torch.load(Path(tmp) / 'run/checkpoint_last.pt', weights_only=False)
            snap[f'train/{method}/model'] = {k: v.detach().cpu().numpy().copy() for k, v in ck['model'].items()}
            snap[f'train/{method}/steps'] = int(ck['steps'])
            rows = [json.loads(x) for x in (Path(tmp) / 'run/log.jsonl').read_text().splitlines()]
            snap[f'train/{method}/log'] = [{k: v for k, v in r.items() if isinstance(v, (int, float)) and
                                            not any(t in k.lower() for t in VOLATILE)} for r in rows]
    return snap


def compare(a, b, path='', diffs=None):
    diffs = [] if diffs is None else diffs
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append(f'{path}/{k}: only in {"before" if k in a else "after"}')
            else:
                compare(a[k], b[k], f'{path}/{k}', diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f'{path}: length {len(a)} vs {len(b)}')
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f'{path}[{i}]', diffs)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
            diffs.append(f'{path}: arrays differ (shape {a.shape} vs {b.shape}, max|diff|='
                         f'{np.abs(a.astype(float) - b.astype(float)).max() if a.shape == b.shape and a.dtype.kind in "fiub" else "n/a"})')
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b)):
        diffs.append(f'{path}: {a!r} vs {b!r}')
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save'); ap.add_argument('--check')
    args = ap.parse_args()
    snap = snapshot()
    if args.save:
        torch.save(snap, args.save); print(f'[OK] snapshot with {len(snap)} groups -> {args.save}')
    if args.check:
        before = torch.load(args.check, weights_only=False)
        for key, rows in before.items():
            if key.endswith('/log'):
                before[key] = [{RENAMED.get(k, k): v for k, v in r.items() if k not in DROPPED} for r in rows]
        diffs = compare(before, snap)
        if diffs:
            print(f'[FAIL] {len(diffs)} difference(s):'); print('\n'.join('  ' + d for d in diffs[:40]))
            raise SystemExit(1)
        print(f'[OK] all {len(snap)} groups identical to {args.check}')


if __name__ == '__main__':
    main()
