"""Persist full clairvoyant CP-SAT reference results; never an online comparator."""
from pathlib import Path
import json
from dataclasses import asdict
from agent.offline import solve_cpsat, replay_check
from environment.problem import Problem
from configs.experiment import environment_config
from data.benchmark import fixtures, prepare
from result.storage import object_hash, digest, atomic_json, write_csv
from result.provenance import evaluation_hash


def evaluate(root,data,c):
    root=Path(root);cases={r['instance_id']:r for r in prepare(c,data)['cases']};rows=[];schedules=[]
    for inst in fixtures(c,data,'small'):
        params=c['experiment']['exact'];ident=object_hash(dict(instance=cases[inst.instance_id]['sha256'],parameters=params,
            solver=evaluation_hash()))
        p=root/'exact'/f'{inst.instance_id}.json'
        if p.exists():
            record=json.loads(p.read_text())
            if record['identity']!=ident: raise ValueError('Offline reference identity mismatch')
        else:
            problem=Problem(inst);out=solve_cpsat(problem,params['wall_seconds'],params['workers'],params['deterministic_seconds'])
            feasible=out.status in ('OPTIMAL','FEASIBLE')
            replay=replay_check(problem,out,environment_config(c)) if feasible else None
            if replay is not None and not replay['match']: raise AssertionError(replay)
            assignments=[dict(order=k[0],stage=k[1],machine=v[0],start=v[1],end=v[2]) for k,v in out.assignment.items()]
            record=dict(identity=ident,instance_id=inst.instance_id,status=out.status,
                        eta=out.eta if feasible else None,upper=out.upper if out.upper==out.upper else None,
                        seconds=out.seconds,assignment=assignments,replay=replay,parameters=params,
                        replay_steps=len(assignments) if feasible else 0,metadata=out.metadata,
                        interpretation='clairvoyant offline reference; only OPTIMAL proves optimality')
            atomic_json(p,record)
        rows.append({k:v for k,v in record.items() if k not in ('assignment','replay','parameters','metadata')})
        schedules.extend(dict(instance_id=inst.instance_id,**r) for r in record['assignment'])
    write_csv(root/'exact.csv',rows);write_csv(root/'exact_schedule.csv',schedules,
        ['instance_id','order','stage','machine','start','end'])
    atomic_json(root/'exact_manifest.json',dict(cases={p.name:digest(p) for p in (root/'exact').glob('*.json')},
        outputs={n:digest(root/n) for n in ('exact.csv','exact_schedule.csv')}))
