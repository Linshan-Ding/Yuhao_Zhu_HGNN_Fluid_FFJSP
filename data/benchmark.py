"""One immutable realization per physical parameter cell (plus independent replicates on the contested grid), shared across all methods."""
from pathlib import Path
import json
import numpy as np
from data.generator import Instance, build_instance, load_metrics, save_instance_csv, load_instance_csv, route_minimum, deadlines, stream_meta
from result.storage import atomic_json, digest, object_hash, write_csv

SPLITS = ('validation', 'main', 'small', 'long_stream', 'large_shop', 'contested_validation', 'contested', 'contested_small', 'sensitivity')
CONTESTED_ROLES = (('contested_validation', 'contested_validation'), ('contested', 'contested'), ('contested_small', 'contested'))


def design(c):
    """Rows of the fixed benchmark: legacy cells are parameterized by arrival intensity (iotas), contested cells by the
    target load (rhos) with independent replicates; both share the due-factor levels of their group."""
    d, b = c['data'], c['benchmark']; rows = []
    shape = dict(products=d['products'], stages=d['stages'], machines_per_stage=d['machines_per_stage'])
    for role, levels in [('validation', b['validation_orders']), ('main', b['main_orders']),
                         ('small', b['small_orders']), ('long_stream', b['long_stream_orders'])]:
        for n in levels:
            rows.append(dict(role=role, family='validation' if role == 'validation' else 'standard', **shape,
                             orders=n, iotas=d['grid_iota'], dues=d['grid_due']))
    for s in b['large_designs']:
        rows.append(dict(role='large_shop', **s, iotas=b['large_iota'], dues=d['grid_due']))
    for role, family in CONTESTED_ROLES:
        g = b[role]
        for n in g['orders']:
            rows.append(dict(role=role, family=family, **shape, orders=n, rhos=g['rhos'], dues=g['dues'], replicates=g['replicates']))
    if c['purpose'] == 'engineering-smoke':
        rows = [dict(role=role, family=family, products=p, stages=s, machines_per_stage=2,
                     orders=n, iotas=[1.5], dues=[3.]) for role, family, p, s, n in
                [('validation','validation',3,3,6),('main','standard',3,3,8),('small','standard',3,3,4),
                 ('long_stream','standard',3,3,10),('large_shop','large5',3,4,12),('large_shop','large7',4,3,14)]]
        rows += [dict(role=role, family=family, products=3, stages=3, machines_per_stage=2, orders=n, rhos=[1.3], dues=[1.5], replicates=2)
                 for (role, family), n in zip(CONTESTED_ROLES, (6, 8, 4))]
    return rows


def cells(rows):
    """(row, iota or None, due, rho or None, replicate or None) for every fixed cell of a design."""
    out = []
    for s in rows:
        if 'rhos' in s:
            out.extend((s, None, due, rho, r) for rho in s['rhos'] for due in s['dues'] for r in range(s['replicates']))
        else:
            out.extend((s, iota, due, None, None) for iota in s['iotas'] for due in s['dues'])
    return out


def parameter_key(c, s, iota, due, rho=None, replicate=None):
    """Legacy cells keep their original key (and therefore their ids); contested cells add the target load and replicate."""
    key = dict(orders=s['orders'], products=s['products'], stages=s['stages'],
               machines_per_stage=[s['machines_per_stage']] * s['stages'], processing=c['data']['processing'],
               eligibility_probability=c['data']['eligibility_probability'], arrival_process='poisson')
    if rho is None: key.update(iota=float(iota), due_factor=float(due))
    else: key.update(rho_target=float(rho), due_factor=float(due), replicate=int(replicate))
    return key


def dataset_identity(c):
    return object_hash(dict(data=c['data'], benchmark=c['benchmark'], design=design(c), format='fixed-cell-3'))


PREFIX = {'validation': 'V', 'main': 'M', 'small': 'S', 'long_stream': 'L', 'large_shop': 'H',
          'contested_validation': 'CV', 'contested': 'C', 'contested_small': 'CS'}


def number(x):
    """Two decimals with trailing zeros removed but at least one kept: 1.0 -> '1.0', 1.15 -> '1.15', 0.6 -> '0.6'."""
    s = f'{float(x):.2f}'.rstrip('0')
    return s + '0' if s.endswith('.') else s


def label(role, row, key):
    """Human-readable, paper-ready instance id built from the design cell: a role prefix (V/M/S/L/H<stages>/CV/C/CS)
    and the order count, then I<iota>-K<due factor> for legacy cells or R<target load>-K<due factor>-r<replicate> for
    contested cells, e.g. M48-I1.5-K3.0, H7-192-I2.5-K1.5, C96-R1.6-K2.0-r3."""
    head = f"H{row['stages']}-{key['orders']}" if role == 'large_shop' else f"{PREFIX[role]}{key['orders']}"
    if 'rho_target' in key:
        return f"{head}-R{number(key['rho_target'])}-K{number(key['due_factor'])}-r{key['replicate'] + 1}"
    return f"{head}-I{number(key['iota'])}-K{number(key['due_factor'])}"


def catalog_intensity(catalog, rho):
    """Arrival intensity (per mean processing time) that puts the catalog at system load `rho`."""
    return float(rho) * catalog.meta['p_bar'] / catalog.meta['rho_sys']


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
        if len(keys) != len(cells(design(c))): raise ValueError('Incomplete or conflicting parameter grid')
        legacy = {object_hash(parameter_key(c, s, iota, due)) for s, iota, due, rho, r in cells(design(c)) if rho is None}
        if not legacy <= set(keys): raise ValueError('Incomplete or conflicting parameter grid')
        for first, second in (('validation', 'main'), ('contested_validation', 'contested')):
            if set(saved['splits'][first]) & set(saved['splits'][second]): raise ValueError('Validation/test overlap')
        if saved['splits']['sensitivity'] != saved['splits']['contested']: raise ValueError('Sensitivity must alias contested')
        return saved
    root.mkdir(parents=True, exist_ok=True); rows = design(c); families = {}; entries = []; files = {}
    for family in dict.fromkeys(s['family'] for s in rows):
        group = [s for s in rows if s['family'] == family]; s = group[0]
        seed = c['benchmark']['seeds'][family]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        intensities = sorted(set(x for r in group for x in r.get('iotas', [])))
        targets = sorted(set(x for r in group for x in r.get('rhos', [])))
        for attempt in range(1, c['data']['max_attempts'] + 1):
            catalog = build_instance(rng, instance_id=family, tier='catalog', product_count=s['products'],
                stage_count=s['stages'], machines_per_stage=[s['machines_per_stage']] * s['stages'], order_count=0,
                proc_time_range=c['data']['processing'], ddt=1, mean_interarrival=1,
                eligibility_prob=c['data']['eligibility_probability'])
            loads = [x * catalog.meta['rho_sys'] / catalog.meta['p_bar'] for x in intensities] + targets
            if all(c['data']['fixed_rho'][0] <= x <= c['data']['fixed_rho'][1] for x in loads): break
        else: raise ValueError(f'No jointly admissible catalogue in {c["data"]["max_attempts"]} attempts: {family}')
        maximum = max(s['orders'] for s in group); replicates = max([s.get('replicates', 1) for s in group])
        route = route_minimum(catalog.proc_times, catalog.product_count, catalog.stage_count)
        families[family] = dict(seed=seed, attempts=attempt, iota=intensities, rho=loads, rho_targets=targets, replicates=replicates,
                               streams={'catalog':0,'products':1,'unit_interarrival':2,'deadline_jitter':3},
                               catalog=object_hash(catalog.proc_times), maximum_orders=maximum)
        for r in range(replicates):
            # One shared order prefix per replicate (legacy families keep their original single-realization streams).
            spawn = (lambda i: np.random.SeedSequence([seed, i])) if 'rhos' not in s else (lambda i: np.random.SeedSequence([seed, i, r]))
            streams = [np.random.default_rng(spawn(i)) for i in (1, 2, 3)]
            products = streams[0].integers(0, catalog.product_count, maximum)
            gaps = streams[1].exponential(1., maximum); gaps[0] = 0.
            jitter = streams[2].uniform(*c['data']['due_jitter'], maximum)
            for row, iota, due, rho, rep in cells(group):
                if rep is not None and rep != r: continue
                if rep is None and r > 0: continue
                intensity = iota if rho is None else catalog_intensity(catalog, rho)
                key = parameter_key(c, row, intensity, due, rho, rep); cell = object_hash(key); name = label(row['role'], row, key)
                if any(e['parameter_hash'] == cell or e['instance_id'] == name for e in entries): raise ValueError('Duplicate parameter configuration')
                n = row['orders']; lam = intensity / catalog.meta['p_bar']
                arrivals = np.cumsum(gaps[:n]) / lam
                due_dates = deadlines(arrivals, route, products[:n], due, jitter[:n])
                extra = {} if rho is None else dict(rho_target=float(rho), replicate=int(rep))   # legacy files stay byte-identical
                meta = stream_meta(c, route, lam, due, intensity, family=family, family_seed=seed, sampling_attempts=attempt,
                                   sampling_regime='joint_fixed_grid' if rho is None else 'target_load_grid', catalog_sha256=families[family]['catalog'],
                                   parameter_hash=cell, **extra)
                inst = Instance(name, row['role'], catalog.product_count, catalog.stage_count, catalog.machines_per_stage,
                                catalog.proc_times.copy(), products[:n].copy(), arrivals, due_dates, meta)
                inst.meta.update(load_metrics(inst)); path = root / 'cases' / f'{name}.csv'
                # An interrupted preparation may leave identical files; never replace conflicting data.
                if path.exists():
                    prior = load_instance_csv(path)
                    if any(not np.array_equal(getattr(prior,k),getattr(inst,k)) for k in
                           ('proc_times','order_product','arrival_times','due_dates')): raise ValueError(f'Conflicting instance: {path}')
                else: save_instance_csv(inst, path)
                relative = path.relative_to(root).as_posix(); files[relative] = digest(path)
                entries.append(dict(instance_id=name, legacy_id='case_' + cell[:16], role=row['role'], family=family, parameter_hash=cell,
                    file=relative, sha256=files[relative], **{k: v for k, v in key.items() if k not in ('iota', 'rho_target', 'replicate')},
                    iota=float(intensity), rho_target=meta.get('rho_target', -1.), replicate=meta.get('replicate', -1), machines=inst.machine_count,
                    catalog_sha256=meta['catalog_sha256'], seed=seed, sampling_attempts=attempt,
                    **{k:inst.meta[k] for k in ('Lambda','empirical_Lambda','p_bar','rho_sys')}))
    splits = {r:[e['instance_id'] for e in entries if e['role']==r] for r in SPLITS}
    splits['sensitivity'] = list(splits['contested'])
    if c['purpose'] != 'engineering-smoke' and len(entries) != c['benchmark']['expected_cells']:
        raise AssertionError(f"Formal benchmark must have {c['benchmark']['expected_cells']} unique cells, got {len(entries)}")
    write_csv(root / 'index.csv', [{**e, 'machines_per_stage':json.dumps(e['machines_per_stage']),
                                  'processing':json.dumps(e['processing'])} for e in entries])
    saved = dict(format='fixed-cell-3', fingerprint=fingerprint, files=files, cases=entries, splits=splits,
                 families=families, index_sha256=digest(root/'index.csv'))
    atomic_json(manifest, saved); return saved


def fixtures(c, root, split):
    split = 'contested_validation' if split == 'monitor' else split
    saved = prepare(c, root); by_id = {r['instance_id']:r for r in saved['cases']}
    return [load_instance_csv(Path(root)/by_id[i]['file'], tier=split, instance_id=i) for i in saved['splits'][split]]


def instance_by_id(c, root, instance_id):
    """One fixed instance by its readable id (single-instance training and evaluation)."""
    saved = prepare(c, root); entry = next((r for r in saved['cases'] if r['instance_id'] == instance_id), None)
    if entry is None: raise KeyError(f'unknown fixed instance: {instance_id}')
    return load_instance_csv(Path(root)/entry['file'], tier=entry['role'], instance_id=instance_id)
