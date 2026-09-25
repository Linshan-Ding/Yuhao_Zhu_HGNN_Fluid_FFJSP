"""Single configuration source and immutable formal job specifications."""
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import yaml

ROOT = Path(__file__).resolve().parents[1]
RULES = ('SPT', 'FCFS', 'EDD', 'MST', 'CR', 'LWKR')
ALL_RULES = (*RULES, 'SPT-mixed')
CLASSIC = ('dqn', 'ddqn', 'a2c', 'ppo')
ABLATIONS = ('no_graph', 'no_scenario', 'no_demo', 'impact_only')
METHODS = ('full', *CLASSIC, 'hgnn', 'dual_attention', *ABLATIONS)


def config(micro=False):
    c = yaml.safe_load((ROOT / 'configs/experiment.yaml').read_text(encoding='utf-8'))
    if micro:
        c['purpose'] = 'engineering-smoke'
        c['data'].update(orders=[6, 8])
        c['network'].update(width=16, layers=1, action_layers=1, candidate_chunk=16)
        c['demonstration'].update(steps=32, epochs=2, minibatch=16)
        c['training'].update(rollout_steps=32, minibatch=16, epochs=1, environments=2,
                             milestones=[64, 128], evaluate_every=128)
        c['scenario'].update(warmup_real_steps=16, teacher_interval=256, replay_batch=4)
        c['classic'].update(width=32, start=16, batch=8, replay_size=512, a2c_rollout=32)
        c['experiment'].update(seeds=[1], budget=128, sensitivity_seeds=[1], sensitivity_budget=128,
                               latency_warmup=1, latency_repeats=2)
        c['experiment']['exact'].update(wall_seconds=2, deterministic_seconds=1)
    return c


def identity(c):
    # Operational paths/concurrency/reporting do not define the training algorithm.
    ignored = {'runtime', 'recording', 'smoke', 'purpose', 'experiment'}
    return hashlib.sha256(json.dumps({k: v for k, v in c.items() if k not in ignored},
                                     sort_keys=True).encode()).hexdigest()


def environment_config(c):
    from configs.config import Config
    return Config({'action_space': {'allow_noop': True, 'wait_interval': c['environment']['wait_interval']},
                   'episode': {'max_decision_steps': c['environment']['max_steps']}})


def uses_demo(method):
    return method in ('full', 'no_graph', 'no_scenario', 'impact_only')


def uses_graph(method):
    return method in ('full', 'no_scenario', 'no_demo', 'impact_only')


def uses_scenario(method):
    return method in ('full', 'no_graph', 'no_demo', 'impact_only')


@dataclass(frozen=True)
class RunSpec:
    name: str
    method: str
    seed: int
    budget: int
    group: str
    config: dict

    def record(self):
        return asdict(self)


def matrix(c):
    e = c['experiment']; jobs = []
    for method in METHODS:
        for seed in e['seeds']:
            d = deepcopy(c); d['method'] = method
            if method in CLASSIC:
                d['training']['learning_rate'] = c['classic']['learning_rates'][method]
            group = 'main' if method == 'full' else 'comparators'
            jobs.append(RunSpec(f'{method}_s{seed}', method, seed, e['budget'], group, d))
    for name, change in e['sensitivity_changes'].items():
        for seed in e['sensitivity_seeds']:
            d = deepcopy(c); d['method'] = 'full'; d['scenario'].update(change)
            d['training']['milestones'] = [x for x in d['training']['milestones'] if x <= e['sensitivity_budget']]
            jobs.append(RunSpec(f'sensitivity_{name}_s{seed}', 'full', seed, e['sensitivity_budget'], 'sensitivity', d))
    if c['purpose'] == 'formal':
        assert len(jobs) == 73 and sum(s.budget for s in jobs) == 64000000
    return jobs
