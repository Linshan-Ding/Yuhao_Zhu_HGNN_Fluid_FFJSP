"""Single configuration source, pre-registered teacher selection and immutable job specifications."""
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
STUDIES = ('formal', 'pilot', 'instance')


def config(micro=False):
    """The single configuration; `micro` applies the engineering-smoke reductions (budget from smoke.budget)."""
    c = yaml.safe_load((ROOT / 'configs/experiment.yaml').read_text(encoding='utf-8'))
    if c['runtime']['device'] != 'cpu':
        raise ValueError(f"runtime.device must be 'cpu' (this study runs CPU-only); got {c['runtime']['device']!r}")
    if micro:
        b = c['smoke']['budget']
        c['purpose'] = 'engineering-smoke'
        c['data'].update(orders=[6, 8])
        c['network'].update(width=16, layers=1)
        c['demonstration'].update(steps=32, epochs=2, minibatch=16)
        c['training'].update(rollout_steps=32, minibatch=16, epochs=1, environments=2,
                             milestones=[b // 2, b], evaluate_every=b)
        c['scenario'].update(warmup_real_steps=16, frozen_interval=256, replay_batch=4)
        c['classic'].update(width=32, start=16, batch=8, replay_size=512, a2c_rollout=32)
        c['experiment'].update(seeds=[1], budget=b, sensitivity_seeds=[1], sensitivity_budget=b,
                               evaluate_milestones_from=b // 2, latency_warmup=1, latency_repeats=2)
        c['experiment']['exact'].update(wall_seconds=2, deterministic_seconds=1)
        c['pilot'].update(seeds=[1], budget=b)
        c['instance_study'].update(methods=['full', 'ppo'], seeds=[1], budget=b, evaluate_every=b // 2, milestones=[b // 2, b], demonstration_steps=32)
        c['teacher'] = 'SPT'   # engineering smoke uses a fixed rule; the selection itself is exercised separately
    return c


def identity(c):
    # Operational paths/concurrency/reporting do not define the training algorithm.
    ignored = {'runtime', 'recording', 'smoke', 'purpose', 'experiment', 'pilot', 'instance_study'}
    return hashlib.sha256(json.dumps({k: v for k, v in c.items() if k not in ignored},
                                     sort_keys=True).encode()).hexdigest()


def environment_config(c):
    from configs.config import Config
    e = c['environment']
    return Config({'action_space': {'allow_noop': True, 'exposure': e['exposure'], 'exposure_threshold': e['exposure_threshold']},
                   'episode': {'max_decision_steps': e['max_steps']}})


def resolve_teacher(c, data):
    """Pre-registered teacher selection on the contested validation split (never on test instances).

    With `teacher: auto`, every rule in RULES is evaluated on the fixed validation cells, the rule with the highest mean
    fulfillment (ties broken by RULES order) is written to <data>/teacher.json and returned in a copy of the
    configuration, so that run identities carry a concrete rule. A named teacher is returned unchanged."""
    if c['teacher'] != 'auto':
        return deepcopy(c)
    from agent.rules import RulePolicy
    from data.benchmark import fixtures, prepare
    from result.evaluation import evaluate_instances
    from result.storage import atomic_json
    import numpy as np
    dataset = prepare(c, data); instances = fixtures(c, data, 'contested_validation')
    scores = {rule: float(np.mean([r['eta'] for r in evaluate_instances(RulePolicy(rule), instances, c)])) for rule in RULES}
    best = max(RULES, key=lambda r: (scores[r], -RULES.index(r)))
    atomic_json(Path(data) / 'teacher.json', dict(rule=best, scores=scores, split='contested_validation',
                instances=[i.instance_id for i in instances], fingerprint=dataset['fingerprint'],
                rule_order=list(RULES), selection='highest mean fulfillment on the contested validation split; ties by rule order'))
    d = deepcopy(c); d['teacher'] = best
    return d


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
    instance: str = None   # single-instance study: the fixed instance this job trains and is evaluated on

    def record(self):
        return asdict(self)


def job(c, method, seed, budget, group):
    d = deepcopy(c); d['method'] = method
    if method in CLASSIC:
        d['training']['learning_rate'] = c['classic']['learning_rates'][method]
    d['training']['milestones'] = [x for x in d['training']['milestones'] if x <= budget]
    return RunSpec(f'{method}_s{seed}', method, seed, budget, group, d)


def matrix(c, study='formal', instances=None):
    """Immutable job list of a study: the formal matrix (73 jobs, 64M interactions), the pre-registered pilot, or the
    single-instance convergence study (every method and seed trained on each fixed instance of `instance_study.split`)."""
    if study not in STUDIES:
        raise ValueError(f'unknown study: {study}')
    if c['teacher'] == 'auto':
        raise ValueError('resolve_teacher must run before the job matrix is built')
    e = c['experiment']; jobs = []
    if study == 'instance':
        s = c['instance_study']
        if not instances:
            raise ValueError('the single-instance study needs the ids of its fixed instances')
        for instance in instances:
            for method in s['methods']:
                for seed in s['seeds']:
                    d = deepcopy(c); d['method'] = method; d['instance'] = instance
                    if method in CLASSIC:
                        d['training']['learning_rate'] = c['classic']['learning_rates'][method]
                    d['training'].update(evaluate_every=s['evaluate_every'], milestones=[x for x in s['milestones'] if x <= s['budget']])
                    d['demonstration']['steps'] = s['demonstration_steps']
                    jobs.append(RunSpec(f'{method}_{instance}_s{seed}', method, seed, s['budget'], 'instance', d, instance))
        return jobs
    if study == 'pilot':
        p = c['pilot']
        for method in p['methods']:
            for seed in p['seeds']:
                jobs.append(job(c, method, seed, p['budget'], 'main' if method == 'full' else 'comparators'))
        return jobs
    for method in METHODS:
        for seed in e['seeds']:
            jobs.append(job(c, method, seed, e['budget'], 'main' if method == 'full' else 'comparators'))
    for name, change in e['sensitivity_changes'].items():
        for seed in e['sensitivity_seeds']:
            d = deepcopy(c); d['method'] = 'full'; d['scenario'].update(change)
            d['training']['milestones'] = [x for x in d['training']['milestones'] if x <= e['sensitivity_budget']]
            jobs.append(RunSpec(f'sensitivity_{name}_s{seed}', 'full', seed, e['sensitivity_budget'], 'sensitivity', d))
    if c['purpose'] == 'formal':
        assert len(jobs) == 73 and sum(s.budget for s in jobs) == 64000000
    return jobs
