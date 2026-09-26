"""Environment-owned physical replay for a clairvoyant offline schedule."""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List
from environment.problem import Problem

Schedule = Dict[Tuple[int, int], Tuple[int, float, float]]   # (order, stage) -> (machine, start, end)


def _replay_in_env(problem: Problem, assignment: Schedule, cfg, tol: float) -> Tuple[int, List[str]]:
    """Physical schedule replay; this is NOT a test of the online policy action set.

    Scheduled starts are external events available to this clairvoyant reference only.
    The environment still validates dispatches and retains all spent processing time.
    """
    from configs.config import Config
    from environment.env import WAITING
    from environment.public import SchedulingEnv

    env = SchedulingEnv(problem.inst, Config(cfg.to_dict()))
    queue = sorted(assignment.items(), key=lambda kv: (kv[1][1], kv[0]))
    issues: List[str] = []
    head = 0
    while not env.done and head < len(queue):
        (s, j), (m, start, _) = queue[head]
        if start < env.now - tol:
            issues.append(f"env: order {s} stage {j} due to start at {start:.4f}, env already at {env.now:.4f}")
            break
        if start > env.now + tol:
            nxt = env._next_event_time()
            env.now = min(start, nxt) if nxt is not None and nxt > env.now + tol else start
            env._activate_arrivals()
            env._release_machines()
            env._discard_hopeless()
            continue
        task = problem.task_of(s, j)
        if env.status[s] != WAITING or int(env.stage[s]) != j:
            issues.append(f"env: order {s} not waiting at stage {j} at {env.now:.4f}")
            break
        if env.machine_busy_with[m] >= 0 or env.machine_free_at[m] > env.now + 1e-9 \
                or problem.rates[task, m] <= 0:
            issues.append(f"env: machine {m} cannot take order {s} stage {j} at {env.now:.4f}")
            break
        env.step((task, m, s))
        head += 1
    if head < len(queue) and not issues:
        issues.append(f"env: episode ended with {len(queue) - head} operations never dispatched")
    while not env.done:                               # 让已派工序全部完工，结清计数
        nxt = env._next_event_time()
        waiting = np.flatnonzero(env.status == WAITING)
        expire = float(env.inst.due_dates[waiting].max()) + 1.0 if len(waiting) else None
        if nxt is None and expire is None:
            env.done = True
            break
        env.now = min(v for v in (nxt, expire) if v is not None and v > env.now)
        env._activate_arrivals(); env._release_machines(); env._discard_hopeless()
    return int(env.n_completed), issues

