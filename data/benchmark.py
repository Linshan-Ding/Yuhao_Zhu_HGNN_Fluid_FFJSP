"""One immutable realization per physical parameter cell, shared across all methods."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from data.generator import Instance, build_instance, load_metrics, save_instance_csv, load_instance_csv
from result.storage import atomic_json, digest, object_hash, write_csv

SPLITS = ('validation', 'main', 'small', 'long_stream', 'large_shop', 'sensitivity')


def design(c):
    d, b = c['data'], c['benchmark']; rows = []
    for role, levels in [('validation', b['validation_orders']), ('main', b['main_orders']),
                         ('small', b['small_orders']), ('long_stream', b['long_stream_orders'])]:
        for n in levels:
            rows.append(dict(role=role, family='validation' if role == 'validation' else 'standard',
                             products=d['products'], stages=d['stages'], machines_per_stage=d['machines_per_stage'],
                             orders=n, iotas=d['grid_iota'], dues=d['grid_due']))
    for s in b['large_designs']:
        rows.append(dict(role='large_shop', **s, iotas=b['large_iota'], dues=d['grid_due']))
    if c['purpose'] == 'engineering-smoke':
        rows = [dict(role=role, family=family, products=p, stages=s, machines_per_stage=2,
                     orders=n, iotas=[1.5], dues=[3.]) for role, family, p, s, n in
                [('validation','validation',3,3,6),('main','standard',3,3,8),('small','standard',3,3,4),
                 ('long_stream','standard',3,3,10),('large_shop','large5',3,4,12),('large_shop','large7',4,3,14)]]
    return rows


def parameter_key(c, s, iota, due):
    return dict(orders=s['orders'], products=s['products'], stages=s['stages'],
                machines_per_stage=[s['machines_per_stage']] * s['stages'], processing=c['data']['processing'],
                eligibility_probability=c['data']['eligibility_probability'], arrival_process='poisson',
                iota=float(iota), due_factor=float(due))


def dataset_identity(c):
    return object_hash(dict(data=c['data'], benchmark=c['benchmark'], design=design(c), format='fixed-cell-1'))


def prepare(c, root):
    root = Path(root); manifest = root / 'manifest.json'; fingerprint = dataset_identity(c)
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved['fingerprint'] != fingerprint: raise ValueError('Fixed data configuration mismatch')
        if digest(root / 'index.csv') != saved['index_sha256']: raise ValueError('Data index changed')
        for name, sha in saved['files'].items():
            if digest(root / name) != sha: raise ValueError(f'Fixed instance changed: {name}')
        keys = [r['parameter_hash'] for r in saved['cases']]
        if len(keys) != len(set(keys)): raise ValueError('Duplicate parameter cells')
        expected={object_hash(parameter_key(c,s,iota,due)) for s in design(c) for iota in s['iotas'] for due in s['dues']}
        if set(keys)!=expected:raise ValueError('Incomplete or conflicting parameter grid')
        if set(saved['splits']['validation']) & set(saved['splits']['main']): raise ValueError('Validation/test overlap')
        if saved['splits']['sensitivity'] != saved['splits']['main']: raise ValueError('Sensitivity must alias main')
        return saved
    root.mkdir(parents=True, exist_ok=True); rows = design(c); families = {}; entries = []; files = {}
    for family in dict.fromkeys(s['family'] for s in rows):
        group = [s for s in rows if s['family'] == family]; s = group[0]
        seed = c['benchmark']['seeds'][family]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        intensities = sorted(set(x for r in group for x in r['iotas']))
        for attempt in range(1, c['data']['max_attempts'] + 1):
            catalog = build_instance(rng, instance_id=family, tier='catalog', product_count=s['products'],
                stage_count=s['stages'], machines_per_stage=[s['machines_per_stage']] * s['stages'], order_count=0,
                proc_time_range=c['data']['processing'], ddt=1, mean_interarrival=1,
                eligibility_prob=c['data']['eligibility_probability'])
            loads = [x * catalog.meta['rho_sys'] / catalog.meta['p_bar'] for x in intensities]
            if all(c['data']['fixed_rho'][0] <= x <= c['data']['fixed_rho'][1] for x in loads): break
        else: raise ValueError(f'No jointly admissible catalogue in {c["data"]["max_attempts"]} attempts: {family}')
        maximum = max(s['orders'] for s in group)
        streams = [np.random.default_rng(np.random.SeedSequence([seed, i])) for i in (1, 2, 3)]
        products = streams[0].integers(0, catalog.product_count, maximum)
        gaps = streams[1].exponential(1., maximum); gaps[0] = 0.
        jitter = streams[2].uniform(.9, 1.1, maximum)
        minimum = np.where(catalog.proc_times > 0, catalog.proc_times, np.inf).min(1).reshape(catalog.product_count, catalog.stage_count).sum(1)
        families[family] = dict(seed=seed, attempts=attempt, iota=intensities, rho=loads,
                               streams={'catalog':0,'products':1,'unit_interarrival':2,'deadline_jitter':3},
                               catalog=object_hash(catalog.proc_times), maximum_orders=maximum)
        for s in group:
            for iota in s['iotas']:
                for due in s['dues']:
                    key = parameter_key(c, s, iota, due); cell = object_hash(key); name = 'case_' + cell[:16]
                    if any(r['parameter_hash'] == cell for r in entries): raise ValueError('Duplicate parameter configuration')
                    n = s['orders']; lam = iota / catalog.meta['p_bar']
                    arrivals = np.cumsum(gaps[:n]) / lam
                    deadlines = arrivals + minimum[products[:n]] * due * jitter[:n]
                    meta = dict(DDT=float(minimum.mean()), mean_interarrival=1/lam, arrival_process='poisson',
                                due_factor=due, iota_target=iota, schema_version=c['schema_version'],
                                family=family, family_seed=seed, sampling_attempts=attempt, sampling_regime='joint_fixed_grid',
                                catalog_sha256=families[family]['catalog'], parameter_hash=cell)
                    inst = Instance(name, s['role'], catalog.product_count, catalog.stage_count, catalog.machines_per_stage,
                                    catalog.proc_times.copy(), products[:n].copy(), arrivals, deadlines, meta)
                    inst.meta.update(load_metrics(inst)); path = root / 'cases' / f'{name}.csv'
                    # An interrupted preparation may leave identical files; never replace conflicting data.
                    if path.exists():
                        prior = load_instance_csv(path)
                        if any(not np.array_equal(getattr(prior,k),getattr(inst,k)) for k in
                               ('proc_times','order_product','arrival_times','due_dates')): raise ValueError(f'Conflicting instance: {path}')
                    else: save_instance_csv(inst, path)
                    relative = path.relative_to(root).as_posix(); files[relative] = digest(path)
                    entries.append(dict(instance_id=name, role=s['role'], family=family, parameter_hash=cell,
                        file=relative, sha256=files[relative], **key, machines=inst.machine_count,
                        catalog_sha256=meta['catalog_sha256'], seed=seed, sampling_attempts=attempt,
                        **{k:inst.meta[k] for k in ('Lambda','empirical_Lambda','p_bar','rho_sys')}))
    splits = {r:[e['instance_id'] for e in entries if e['role']==r] for r in SPLITS}
    splits['sensitivity'] = list(splits['main'])
    if c['purpose'] == 'formal' and len(entries) != 50: raise AssertionError('Formal benchmark must have 50 unique cells')
    write_csv(root / 'index.csv', [{**e, 'machines_per_stage':json.dumps(e['machines_per_stage']),
                                  'processing':json.dumps(e['processing'])} for e in entries])
    saved = dict(format='fixed-cell-1', fingerprint=fingerprint, files=files, cases=entries, splits=splits,
                 families=families, index_sha256=digest(root/'index.csv'))
    atomic_json(manifest, saved); return saved


def fixtures(c, root, split):
    split = 'validation' if split == 'monitor' else split
    saved = prepare(c, root); by_id = {r['instance_id']:r for r in saved['cases']}
    return [load_instance_csv(Path(root)/by_id[i]['file'], tier=split, instance_id=i) for i in saved['splits'][split]]
