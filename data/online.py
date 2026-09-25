"""Frequency-conditioned online generation and immutable, indexed evaluation cohorts."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from data.generator import Instance, build_instance, sample_arrivals, load_metrics, save_instance_csv, load_instance_csv

def sample(cfg, rng, name='training', tier='train', iota=None, due_factor=None):
    d=cfg['data']
    regime = d['regimes'][int(rng.choice(len(d['regimes']),p=[r['weight'] for r in d['regimes']]))] if iota is None else None
    for attempt in range(1,d['max_attempts']+1):
        catalog=build_instance(rng,instance_id=name,tier=tier,product_count=d['products'],stage_count=d['stages'],
            machines_per_stage=[d['machines_per_stage']]*d['stages'],order_count=0,proc_time_range=d['processing'],
            ddt=1,mean_interarrival=1,eligibility_prob=d['eligibility_probability'])
        intensity=float(rng.uniform(*regime['iota'])) if regime else float(iota)
        pbar=catalog.meta['p_bar']; lam=intensity/pbar
        rho=lam*catalog.meta['rho_sys']  # catalog uses unit arrival rate
        lo,hi=regime['rho'] if regime else d['fixed_rho']
        if rho < lo or rho > hi or (regime and regime['upper_open'] and rho >= hi): continue
        count=int(rng.integers(d['orders'][0],d['orders'][1]+1))
        products=rng.integers(0,d['products'],size=count)
        arrivals=sample_arrivals(rng,count,1/lam)
        minimum=np.where(catalog.proc_times>0,catalog.proc_times,np.inf).min(1).reshape(d['products'],d['stages']).sum(1)
        due=float(rng.uniform(*d['due_factor'])) if due_factor is None else float(due_factor)
        deadlines=arrivals+minimum[products]*due*rng.uniform(.9,1.1,count)
        meta={'DDT':float(minimum.mean()),'mean_interarrival':1/lam,'arrival_process':'poisson',
              'due_factor':due,'iota_target':intensity,'schema_version':cfg['schema_version'],
              'sampling_attempts':attempt,'sampling_regime':regime['name'] if regime else 'fixed',
              'catalog_sha256':hashlib.sha256(catalog.proc_times.tobytes()).hexdigest()}
        inst=Instance(name,tier,d['products'],d['stages'],catalog.machines_per_stage,catalog.proc_times,
                      products,arrivals,deadlines,meta)
        inst.meta.update(load_metrics(inst))
        return inst
    raise ValueError(f'No admissible frequency/load combination after {d["max_attempts"]} attempts: {regime or iota}')
