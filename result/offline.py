"""Persist full clairvoyant CP-SAT reference results; never an online comparator."""
from pathlib import Path
import json
from environment.offline_cpsat import solve_cpsat, replay_check
from environment.problem import Problem
from configs.experiment import environment_config
from data.benchmark import fixtures, prepare
from result.storage import object_hash, digest, atomic_json, write_csv
from result.provenance import evaluation_hash


def reference_identity(instance_sha,params):
    return object_hash(dict(instance=instance_sha,parameters=params,solver=evaluation_hash()))


def solve_case(inst,params,c,split):
    """Solve one fixed instance and return the persisted record body (without its identity)."""
    problem=Problem(inst);out=solve_cpsat(problem,params['wall_seconds'],params['workers'],params['deterministic_seconds'])
    feasible=out.status in ('OPTIMAL','FEASIBLE')
    replay=replay_check(problem,out,environment_config(c)) if feasible else None
    if replay is not None and not replay['match']: raise AssertionError(replay)
    assignments=[dict(order=k[0],stage=k[1],machine=v[0],start=v[1],end=v[2]) for k,v in out.assignment.items()]
    return dict(instance_id=inst.instance_id,split=split,status=out.status,eta=out.eta if feasible else None,
                upper=out.upper if out.upper==out.upper else None,seconds=out.seconds,assignment=assignments,replay=replay,
                parameters=params,replay_steps=len(assignments) if feasible else 0,metadata=out.metadata,
                interpretation='clairvoyant offline reference; only OPTIMAL proves optimality, otherwise the bound caps every policy')


def reference(root,inst,instance_sha,c,split,reuse=()):
    """The persisted record of one instance: an existing record with the same identity, a copy of a record with the
    same identity from a `reuse` root (another study on the same instance), or a fresh solve."""
    root=Path(root);params=c['experiment']['exact'];ident=reference_identity(instance_sha,params);p=root/'exact'/f'{inst.instance_id}.json'
    if p.exists():
        record=json.loads(p.read_text())
        if record['identity']!=ident: raise ValueError('Offline reference identity mismatch')
        return record
    for other in reuse:
        q=Path(other)/'exact'/f'{inst.instance_id}.json'
        if q.exists():
            record=json.loads(q.read_text())
            if record['identity']==ident:
                record=dict(record,reused_from=Path(other).as_posix());atomic_json(p,record);return record
    record=dict(identity=ident,**solve_case(inst,params,c,split));atomic_json(p,record);return record


def evaluate(root,data,c,splits=None,reuse=()):
    root=Path(root);cases={r['instance_id']:r for r in prepare(c,data)['cases']};rows=[];schedules=[]
    for split in splits or c['experiment']['exact_splits']:
        for inst in fixtures(c,data,split):
            record=reference(root,inst,cases[inst.instance_id]['sha256'],c,split,reuse)
            rows.append({k:v for k,v in record.items() if k not in ('assignment','replay','parameters','metadata')})
            schedules.extend(dict(instance_id=inst.instance_id,**r) for r in record['assignment'])
    write_csv(root/'exact.csv',rows);write_csv(root/'exact_schedule.csv',schedules,
        ['instance_id','order','stage','machine','start','end'])
    atomic_json(root/'exact_manifest.json',dict(cases={p.name:digest(p) for p in (root/'exact').glob('*.json')},
        outputs={n:digest(root/n) for n in ('exact.csv','exact_schedule.csv')}))
