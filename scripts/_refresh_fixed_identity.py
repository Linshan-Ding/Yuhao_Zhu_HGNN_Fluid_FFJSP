"""Recompute the content identities of the fixed benchmark after the hashing rule changed.

    python scripts/_refresh_fixed_identity.py

Only hashes are rewritten. Every case CSV is parsed before and after, and the script refuses to
proceed if any instance array or manifest parameter would change, so no sample is re-drawn.
The new index.csv is written with LF line endings (result.storage.write_csv).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402

from configs.experiment import config  # noqa: E402
from data.benchmark import dataset_identity, parameter_key, design  # noqa: E402
from data.generator import load_instance_csv  # noqa: E402
from result.storage import atomic_json, digest, object_hash, read_csv, write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'instances' / 'fixed'
FIELDS = ('proc_times', 'order_product', 'arrival_times', 'due_dates')


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    index = read_csv(ROOT / 'index.csv')
    before = {r['file']: load_instance_csv(ROOT / r['file']) for r in index}
    changed = []
    for row, entry in zip(index, manifest['cases']):
        assert row['file'] == entry['file'] and row['instance_id'] == entry['instance_id']
        new = digest(ROOT / row['file'])
        if new != row['sha256']:
            changed.append((row['file'], row['sha256'][:12], new[:12]))
        row['sha256'] = entry['sha256'] = manifest['files'][row['file']] = new
    write_csv(ROOT / 'index.csv', index, list(index[0].keys()))
    manifest['index_sha256'] = digest(ROOT / 'index.csv')
    for r in index:  # content must be byte-for-byte identical in every array
        after = load_instance_csv(ROOT / r['file'])
        for f in FIELDS:
            if not np.array_equal(getattr(before[r['file']], f), getattr(after, f)):
                raise SystemExit(f'[FAIL] instance content changed: {r["file"]} {f}')
    # The configuration fingerprint may change when a generation parameter becomes explicit in the YAML
    # (e.g. data.due_jitter); the parameter grid itself must be exactly the recorded one.
    c = config(); grid = {object_hash(parameter_key(c, s, iota, due)) for s in design(c) for iota in s['iotas'] for due in s['dues']}
    if grid != {e['parameter_hash'] for e in manifest['cases']}:
        raise SystemExit('[FAIL] parameter grid differs from the recorded cases')
    fingerprint = dataset_identity(c)
    if fingerprint != manifest['fingerprint']:
        print(f'  fingerprint: {manifest["fingerprint"][:12]}... -> {fingerprint[:12]}... (configuration keys changed; data unchanged)')
        manifest['fingerprint'] = fingerprint
    atomic_json(ROOT / 'manifest.json', manifest)
    print(f'[OK] {len(changed)} of {len(index)} case digests updated; index_sha256={manifest["index_sha256"][:12]}...')
    for name, old, new in changed[:3]:
        print(f'  {name}: {old}... -> {new}...')


if __name__ == '__main__':
    main()
