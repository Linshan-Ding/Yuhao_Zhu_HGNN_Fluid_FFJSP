"""Frequency-conditioned online instance generation for training (evaluation cases are fixed in data.benchmark)."""
import hashlib
from data.generator import Instance, build_instance, sample_arrivals, load_metrics, route_minimum, deadlines, stream_meta

def sample(cfg, rng, name='training', tier='train', iota=None, due_factor=None):
    """A regime draws either an arrival intensity (checked against its load window) or, with `by_rho`, a target load
    whose intensity is derived from the catalog (checked against its intensity window); every regime carries its own
    due-factor window. An explicit `iota` samples the configured fixed load window with the grid due-factor range."""
    d=cfg['data']
    regime = d['regimes'][int(rng.choice(len(d['regimes']),p=[r['weight'] for r in d['regimes']]))] if iota is None else None
    for attempt in range(1,d['max_attempts']+1):
        catalog=build_instance(rng,instance_id=name,tier=tier,product_count=d['products'],stage_count=d['stages'],
            machines_per_stage=[d['machines_per_stage']]*d['stages'],order_count=0,proc_time_range=d['processing'],
            ddt=1,mean_interarrival=1,eligibility_prob=d['eligibility_probability'])
        pbar=catalog.meta['p_bar'];unit=catalog.meta['rho_sys']  # catalog uses unit arrival rate
        if regime and regime.get('by_rho'):
            rho=float(rng.uniform(*regime['rho']));intensity=rho*pbar/unit
            if not regime['iota'][0]<=intensity<=regime['iota'][1]: continue
        else:
            intensity=float(rng.uniform(*regime['iota'])) if regime else float(iota)
            rho=intensity/pbar*unit
            lo,hi=regime['rho'] if regime else d['fixed_rho']
            if rho < lo or rho > hi or (regime and regime['upper_open'] and rho >= hi): continue
        lam=intensity/pbar
        count=int(rng.integers(d['orders'][0],d['orders'][1]+1))
        products=rng.integers(0,d['products'],size=count)
        arrivals=sample_arrivals(rng,count,1/lam)
        route=route_minimum(catalog.proc_times,d['products'],d['stages'])
        window=regime['due'] if regime else (min(d['grid_due']),max(d['grid_due']))
        due=float(rng.uniform(*window)) if due_factor is None else float(due_factor)
        due_dates=deadlines(arrivals,route,products,due,rng.uniform(*d['due_jitter'],count))
        meta=stream_meta(cfg,route,lam,due,intensity,sampling_attempts=attempt,sampling_regime=regime['name'] if regime else 'fixed',
                         catalog_sha256=hashlib.sha256(catalog.proc_times.tobytes()).hexdigest())
        inst=Instance(name,tier,d['products'],d['stages'],catalog.machines_per_stage,catalog.proc_times,
                      products,arrivals,due_dates,meta)
        inst.meta.update(load_metrics(inst))
        return inst
    raise ValueError(f'No admissible frequency/load combination after {d["max_attempts"]} attempts: {regime or iota}')
