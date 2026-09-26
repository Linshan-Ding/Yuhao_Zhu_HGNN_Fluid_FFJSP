"""Process-isolated engineering accounting; no lifetime cap or formal-budget changes."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import time
import uuid

from configs.experiment import ROOT
from result.storage import atomic_json, digest, run_lock

DEFAULT_ROOT = ROOT / 'result/engineering'


@contextmanager
def accounting_lock(root):
    deadline = time.monotonic() + 30
    while True:
        lock = run_lock(Path(root) / 'accounting_owner')
        try:
            lock.__enter__()
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.02)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def summarize(root=DEFAULT_ROOT):
    """Only the summary is shared. Each simulation process owns a unique durable shard."""
    root = Path(root)
    with accounting_lock(root):
        path = root / 'interaction_ledger.json'
        baseline = root / 'accounting/legacy.json'
        if not baseline.exists():
            old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            if old.get('schema') == 'engineering-accounting-2':
                raise ValueError('Missing engineering accounting history')
            baseline.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                temporary=baseline.with_suffix('.tmp')
                with temporary.open('wb') as f:
                    f.write(path.read_bytes());f.flush();os.fsync(f.fileno())
                os.replace(temporary,baseline)
            else:
                atomic_json(baseline, dict(charged_upper_bound=0, committed=0, sessions=[]))
        old = json.loads(baseline.read_text(encoding='utf-8'))
        sessions = [json.loads(p.read_text(encoding='utf-8'))
                    for p in sorted((root / 'accounting/sessions').glob('*.json'))]
        committed = old.get('committed', 0) + sum(s['committed'] for s in sessions)
        charged = old.get('charged_upper_bound', 0) + sum(s['charged_upper_bound'] for s in sessions)
        result = dict(schema='engineering-accounting-2', limit=None,
                      legacy_file='accounting/legacy.json', legacy_sha256=digest(baseline),
                      historical_charged=old.get('charged_upper_bound', 0),
                      committed=committed, charged_upper_bound=charged,
                      uncommitted_upper_bound=charged-committed, sessions=sessions)
        atomic_json(path, result)
        return result


class InteractionCounter:
    def __init__(self, root=DEFAULT_ROOT, label='engineering'):
        self.root = Path(root)
        self.path = self.root / 'accounting/sessions' / f'{os.getpid()}_{uuid.uuid4().hex}.json'
        self.label = label
        self.spent = self.reserved = 0
        self.began = time.time()
        self.closed = False
        self.persist()

    def persist(self):
        atomic_json(self.path, dict(label=self.label, pid=os.getpid(), started=self.began,
                    ended=time.time() if self.closed else None, closed=self.closed,
                    charged_upper_bound=self.reserved, committed=self.spent))

    def charge(self):
        if self.closed:
            raise RuntimeError('Cannot charge a closed engineering session')
        if self.spent >= self.reserved:
            self.reserved = self.spent + 128
            self.persist()
        self.spent += 1

    def close(self):
        if not self.closed:
            self.closed = True
            self.reserved = self.spent
            self.persist()
        return summarize(self.root)


@contextmanager
def counted(label, root=DEFAULT_ROOT):
    from environment.accounting import install, current
    previous = current()
    counter = InteractionCounter(root, label)
    install(counter)
    try:
        yield counter
    finally:
        install(previous)
        counter.close()
