"""Print the content identities recorded in docs/acceptance.json, recomputed from this checkout.

    python scripts/_identity_report.py            # print
    python scripts/_identity_report.py --write    # also update the four identity fields of docs/acceptance.json

All four values are line-ending independent (result.storage.normalized_bytes), so they reproduce on any
platform from the same commit.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402

from configs.experiment import ROOT  # noqa: E402
from result.provenance import evaluation_hash, source_hash  # noqa: E402
from result.storage import digest  # noqa: E402


def identities():
    from scripts.pipeline import smoke_identity
    return dict(training_source=source_hash(), evaluation_source=evaluation_hash(),
                data_manifest_sha256=digest(ROOT / 'data/instances/fixed/manifest.json'),
                smoke_identity=smoke_identity())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--write', action='store_true'); args = ap.parse_args()
    values = identities()
    path = ROOT / 'docs' / 'acceptance.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    for k, v in values.items():
        print(f'{k:22s} {v}  {"(matches docs/acceptance.json)" if saved.get(k) == v else "(differs)"}')
    if args.write:
        saved.update(values)
        path.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'[OK] wrote {path}')


if __name__ == '__main__':
    main()
