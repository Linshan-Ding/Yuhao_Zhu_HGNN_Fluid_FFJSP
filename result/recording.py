"""Lossless event records, chunk commits and content-addressed simulation inputs."""
from dataclasses import asdict, is_dataclass
from pathlib import Path
import gzip
import json
import time
import uuid
import numpy as np
from data.generator import save_instance_csv
from result.storage import atomic_json, digest, disk_check, object_hash, plain

DYNAMIC = ('status', 'stage', 'machine_free_at', 'machine_busy_with', 'machine_busy_time', 'order_outcome')
PHASES = ('startup', 'arrivals', 'drain')


def phase_at(now, last_arrival, startup):
    """Post-hoc phase label: 'startup' before `startup` of the last arrival time, 'drain' after the last arrival."""
    return 'drain' if now >= last_arrival else ('startup' if now < startup * last_arrival else 'arrivals')


def split_time(start, end, last_arrival, startup):
    cuts = sorted(set([start, end] + [t for t in (startup*last_arrival, last_arrival) if start < t < end]))
    return [(phase_at((a+b)/2, last_arrival, startup), b-a) for a,b in zip(cuts[:-1],cuts[1:])]


class Recorder:
    def __init__(self, root, cfg, state=None):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True); self.cfg = cfg
        self.buffers = {}; self.chunks = []; self.episodes = 0; self.pending = {}; self.previous = {}
        self.io_seconds = 0.; self.steps = 0; self.instances = {}; self.artifacts = {}
        if state:
            self.chunks = list(state['chunks']); self.episodes = state['episodes']; self.steps = state['steps']
            self.instances = dict(state['instances']); self.artifacts = dict(state.get('artifacts',{}))
            self.io_seconds = state['io_seconds']
            for r in self.chunks:
                if digest(self.root/r['file']) != r['sha256']: raise ValueError('Recorded chunk corrupted')
            committed = {r['file'] for r in self.chunks}
            orphaned = [p.relative_to(self.root).as_posix() for p in self.root.glob('chunks/*/*.gz')
                        if p.relative_to(self.root).as_posix() not in committed]
            if orphaned: atomic_json(self.root/f'uncommitted_{time.time_ns()}.json', dict(files=orphaned, reason='not in durable checkpoint'))

    def emit(self, stream, row):
        start = time.perf_counter()
        line = json.dumps(row, default=plain, ensure_ascii=False, separators=(',',':'))
        self.buffers.setdefault(stream, []).append(line)
        self.io_seconds += time.perf_counter()-start
        if len(self.buffers[stream]) >= self.cfg['recording']['chunk_rows']: self.flush(stream)

    def flush(self, stream=None):
        start = time.perf_counter()
        for name in ([stream] if stream else list(self.buffers)):
            rows = self.buffers.get(name, [])
            if not rows: continue
            disk_check(self.root, reserve_gb=self.cfg['recording']['minimum_free_gb'])
            p = self.root/'chunks'/name/f'{len(self.chunks):08d}_{uuid.uuid4().hex[:16]}.jsonl.gz'; p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix('.tmp')
            with gzip.open(tmp, 'wt', encoding='utf-8') as f: f.write('\n'.join(rows)+'\n')
            tmp.replace(p)
            self.chunks.append(dict(stream=name, file=p.relative_to(self.root).as_posix(), sha256=digest(p), rows=len(rows), bytes=p.stat().st_size))
            self.buffers[name] = []
        self.io_seconds += time.perf_counter()-start

    def snapshot(self):
        self.flush()
        return dict(chunks=self.chunks.copy(), episodes=self.episodes, steps=self.steps, instances=self.instances.copy(),
                    artifacts=self.artifacts.copy(), io_seconds=self.io_seconds)

    def commit(self, state):
        atomic_json(self.root/'manifest.json', state)

    def attach(self, env, episode_id, kind):
        start = time.perf_counter()
        if episode_id is None:
            self.episodes += 1; episode_id = f'{kind}_{self.episodes:08d}'
        key = object_hash(asdict(env.inst)); p = self.root/'instances'/key[:2]/f'{key[:32]}.csv'
        if key not in self.instances:
            if not p.exists(): save_instance_csv(env.inst, p)
            self.instances[key] = dict(file=p.relative_to(self.root).as_posix(), sha256=digest(p))
        self.io_seconds += time.perf_counter()-start
        self.emit('episodes', dict(episode_id=episode_id, kind=kind, instance=key, now=env.now,
                  step=env.step_count, initial={k:getattr(env,k).copy() for k in DYNAMIC},
                  events=env.take_events(), completed=env.n_completed, discarded=env.n_discarded))
        return episode_id

    def annotate(self, env, **values):
        self.pending[env._record_id] = values

    def before(self, env, action):
        self.previous[env._record_id] = dict(now=env.now, step=env.step_count,
            dynamic={k:getattr(env,k).copy() for k in DYNAMIC}, held=env.stats.held_time,
            legal_actions=[dict(task=t,machine=m,order=o) if t>=0 else dict(wait=True) for t,m,o in env.candidate_actions()],
            waiting_orders=int(np.sum(env.status==1)),busy_machines=int(np.sum(env.machine_busy_with>=0)))

    def after(self, env, action, result, seconds):
        before = self.previous.pop(env._record_id); reward, done, info = result; self.steps += 1
        delta = {}
        for k, old in before['dynamic'].items():
            new = getattr(env,k); ix = np.flatnonzero(new != old)
            delta[k] = dict(index=ix, value=new[ix])
        act = asdict(action) if is_dataclass(action) else {'tuple':list(action)}
        if not act: act = {'wait':True}
        events = env.take_events()
        self.emit('decisions', dict(episode_id=env._record_id, step=env.step_count,
                  start=before['now'], end=env.now, action=act, reward=reward, done=done, info=info,
                  legal_actions=before['legal_actions'],waiting_orders=before['waiting_orders'],busy_machines=before['busy_machines'],
                  phase=phase_at(before['now'],float(env.inst.arrival_times[-1]),self.cfg['recording']['phase_startup_fraction']),
                  delta=delta, held_machine_time=env.stats.held_time-before['held'], environment_seconds=seconds,
                  **self.pending.pop(env._record_id, {})))
        for event in events: self.emit('events', dict(episode_id=env._record_id, decision=env.step_count, **event))
        if done: self.outcomes(env, 'truncated' if env.truncated else 'resolved')

    def outcomes(self, env, reason):
        for order in range(env.inst.order_count):
            self.emit('orders', dict(episode_id=env._record_id, order=order, product=int(env.inst.order_product[order]),
                arrival=float(env.inst.arrival_times[order]), due=float(env.inst.due_dates[order]),
                outcome=int(env.order_outcome[order]), status=int(env.status[order]), stage=int(env.stage[order]),
                snapshot_time=env.now, episode_reason=reason))

    def artifact(self, value, kind):
        # Immutable frozen-policy/public-state snapshots are referenced by the hash of their serialized bytes.
        start = time.perf_counter()
        import io, torch
        buffer = io.BytesIO(); torch.save(value, buffer)
        import hashlib
        raw=buffer.getvalue(); key=hashlib.sha256(raw).hexdigest(); p=self.root/'artifacts'/kind/f'{key[:32]}.pt'
        if key not in self.artifacts:
            p.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists():
                tmp=p.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(p)
            elif digest(p)!=key:raise ValueError('Artifact collision or corruption')
            self.artifacts[key]=dict(file=p.relative_to(self.root).as_posix(),sha256=digest(p))
        self.io_seconds += time.perf_counter()-start
        return key


def records(root, stream):
    root=Path(root); state=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for chunk in state['chunks']:
        if chunk['stream'] != stream: continue
        p=root/chunk['file']
        if digest(p)!=chunk['sha256']: raise ValueError(f'Corrupt raw chunk: {p}')
        with gzip.open(p,'rt',encoding='utf-8') as f:
            for line in f: yield json.loads(line)


def verify(root):
    root=Path(root); state=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for r in [*state['chunks'],*state['instances'].values(),*state.get('artifacts',{}).values()]:
        if digest(root/r['file'])!=r['sha256']: raise ValueError('Raw artifact checksum mismatch')
    if sum(r['rows'] for r in state['chunks'] if r['stream']=='decisions') != state['steps']:
        raise ValueError('Recorded decision count mismatch')
    return state
