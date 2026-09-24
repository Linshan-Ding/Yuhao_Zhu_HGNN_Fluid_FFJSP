"""采样 worker：每个进程一份环境与一份 CPU 策略网络，整条 episode 采完再发回主进程。

所有函数与类都在模块顶层，进程间只传 Python 标量、dict 与 numpy 数组，兼容 Windows 的
spawn 启动方式（Linux 上也强制 spawn，走同一条代码路径）。
"""
from __future__ import annotations

import multiprocessing as mp
import multiprocessing.connection as mpc
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from agent.baselines.rules import rank, select
from agent.buffer import EpisodeRecord
from configs.config import ROOT, Config
from data.generator import load_instance_csv, sample_training_instance
from environment.env import SchedulingEnv, is_noop

METHODS = ("coh", "rule_ppo", "dqn", "hdqn")


# ------------------------------------------------------------------ worker 端策略
class _TorchPolicy:
    def __init__(self, net) -> None:
        import torch
        self.torch = torch
        self.net = net.eval()

    def load_numpy(self, state: Dict[str, np.ndarray]) -> None:
        self.net.load_state_dict({k: self.torch.from_numpy(np.array(v)) for k, v in state.items()})

    def _forward(self, obs):
        from agent.batch import single_batch
        with self.torch.no_grad():
            return self.net(single_batch(obs))


class CoHPolicy(_TorchPolicy):
    """主方法：候选打分 + 门控。act 返回 (执行的候选下标, 记录的动作, log p, value, extras)。"""

    def __init__(self, cfg) -> None:
        from agent.networks import ActorCritic
        super().__init__(ActorCritic(cfg))

    def act(self, obs, actions, env, rng, bp, greedy):
        from agent.networks import greedy_actions
        out = self._forward(obs)
        if greedy:
            a = int(greedy_actions(out)[0].item())
        else:
            p = out.logp[0].exp().double().numpy()
            p = p / p.sum()
            a = int(rng.choice(p.size, p=p))
        return a, a, float(out.logp[0, a].item()), float(out.value[0].item()), {}


class RulePPOPolicy(_TorchPolicy):
    def __init__(self, cfg) -> None:
        from agent.baselines.rule_ppo import RulePolicyNet
        super().__init__(RulePolicyNet(cfg))

    def act(self, obs, actions, env, rng, bp, greedy):
        from agent.baselines.rule_ppo import HOLD, RULE_POOL
        out = self._forward(obs)
        if greedy:
            r = int(out.logp[0].argmax().item())
        else:
            p = out.logp[0].exp().double().numpy()
            p = p / p.sum()
            r = int(rng.choice(p.size, p=p))
        if r == HOLD:
            a = len(actions) - 1                          # no-op 总在末尾
        else:
            a = select(RULE_POOL[r], env, actions, rng)
        return a, r, float(out.logp[0, r].item()), float(out.value[0].item()), {}


class DQNPolicy(_TorchPolicy):
    def __init__(self, cfg) -> None:
        from agent.baselines.dqn_nets import TripletQNet
        super().__init__(TripletQNet(cfg))

    def act(self, obs, actions, env, rng, bp, greedy):
        out = self._forward(obs)
        q = out.q[0].numpy()
        if not greedy and rng.random() < float(bp.get("epsilon", 0.0)):
            a = int(rng.integers(len(actions)))
        else:
            a = int(np.argmax(q))
        return a, a, 0.0, float(q[a]), {}


class HierDQNPolicy(_TorchPolicy):
    def __init__(self, cfg) -> None:
        from agent.baselines.dqn_nets import HierQNet
        super().__init__(HierQNet(cfg))

    def act(self, obs, actions, env, rng, bp, greedy):
        from agent.baselines.rule_ppo import HOLD, RULE_POOL
        out = self._forward(obs)
        eps = 0.0 if greedy else float(bp.get("epsilon", 0.0))
        q_u = out.q_upper[0].numpy()
        valid_u = np.isfinite(q_u)
        if rng.random() < eps:
            r = int(rng.choice(np.flatnonzero(valid_u)))
        else:
            r = int(np.argmax(q_u))
        window = np.zeros(len(actions), bool)
        if r == HOLD:
            window[len(actions) - 1] = True
        else:
            for i in rank(RULE_POOL[r], env, actions, rng)[:3]:
                window[i] = True
        idx = np.flatnonzero(window)
        if rng.random() < eps:
            a = int(rng.choice(idx))
        else:
            q = out.q[0].numpy()
            a = int(idx[np.argmax(q[idx])])
        return a, a, 0.0, float(out.q[0, a].item()), {"rule_action": np.int64(r), "window_mask": window}


POLICIES = {"coh": CoHPolicy, "rule_ppo": RulePPOPolicy, "dqn": DQNPolicy, "hdqn": HierDQNPolicy}


# ------------------------------------------------------------------ episode
def run_episode(env: SchedulingEnv, policy, rng: Optional[np.random.Generator], bp: dict,
                greedy: bool, record: bool = True) -> Optional[EpisodeRecord]:
    steps: List[dict] = []
    while not env.done:
        actions = env.candidate_actions()
        if not actions:
            break
        obs = env.observation(actions)
        a, rec_a, logp, value, extra = policy.act(obs, actions, env, rng, bp, greedy)
        reward, done, info = env.step(actions[a])
        if record:
            steps.append({"obs": obs, "action": rec_a, "env_action": a, "logp": logp, "value": value,
                          "reward": reward, "done": done, "order": info["order"],
                          "hold_id": info["hold_id"], "extra": extra})
        if done:
            break
    if not record or not steps:
        return None
    return EpisodeRecord.from_steps(steps, env)


def episode_rng(run_seed: int, epoch: int, k: int) -> np.random.Generator:
    """episode (epoch, k) 的随机源，与 worker 编号无关：同种子的 run 采到同一批算例与动作。"""
    return np.random.default_rng(np.random.SeedSequence([int(run_seed), int(epoch), int(k)]))


def worker_main(worker_id: int, cfg_dict: dict, method: str, conn, run_seed: int, val_meta: list) -> None:
    import torch
    torch.set_num_threads(1)
    cfg = Config(cfg_dict)
    policy = POLICIES[method](cfg)
    param_table = cfg.get("param_table")
    val = {m["instance_id"]: load_instance_csv(ROOT / m["path"], m["tier"], m["instance_id"])
           for m in val_meta}
    while True:
        msg = conn.recv()
        kind = msg[0]
        if kind == "weights":
            policy.load_numpy(msg[1])
        elif kind == "episode":
            _, epoch, k, bp = msg
            rng = episode_rng(run_seed, epoch, k)
            env = SchedulingEnv(sample_training_instance(rng, param_table), cfg)
            t0 = time.perf_counter()
            rec = run_episode(env, policy, rng, bp, greedy=False)
            rec.info["seconds"] = time.perf_counter() - t0
            conn.send(("episode", k, rec.to_dict()))
        elif kind == "validate":
            env = SchedulingEnv(val[msg[1]], cfg)
            run_episode(env, policy, np.random.default_rng(0), {}, greedy=True, record=False)
            conn.send(("val", msg[1], {"eta": env.eta, "nu": env.nu, "held_share": env.held_share,
                                        "steps": env.step_count,
                                        "noop_rate": env.stats.noop_used / max(env.stats.noop_offered, 1)}))
        elif kind == "close":
            return


# ------------------------------------------------------------------ 主进程侧
class WorkerPool:
    def __init__(self, n_workers: int, cfg_dict: dict, method: str, run_seed: int,
                 val_meta: Sequence[dict]) -> None:
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(int(n_workers)):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=worker_main,
                               args=(w, cfg_dict, method, child, int(run_seed), list(val_meta)),
                               daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)
        self.val_ids = [m["instance_id"] for m in val_meta]

    @staticmethod
    def default_workers() -> int:
        return max(1, (os.cpu_count() or 2) - 2)

    def broadcast(self, state: Dict[str, np.ndarray]) -> None:
        for c in self.conns:
            c.send(("weights", state))

    def _dispatch(self, jobs: List[tuple], expect: str) -> Dict[object, object]:
        """把任务动态分给空闲 worker，直到全部回收。返回 {任务键: 结果}。"""
        pending = list(jobs)
        inflight: Dict[object, object] = {}
        results: Dict[object, object] = {}
        for c in self.conns:
            if pending:
                job = pending.pop(0)
                c.send(job[0])
                inflight[c] = job[1]
        while inflight:
            for c in mpc.wait(list(inflight)):
                try:
                    reply = c.recv()
                except EOFError as exc:
                    self.close()
                    raise RuntimeError("a rollout worker died") from exc
                assert reply[0] == expect, reply[0]
                results[inflight.pop(c)] = reply[2]
                if pending:
                    job = pending.pop(0)
                    c.send(job[0])
                    inflight[c] = job[1]
        return results

    def collect(self, epoch: int, n_episodes: int, bp: dict) -> List[EpisodeRecord]:
        jobs = [(("episode", int(epoch), k, dict(bp)), k) for k in range(int(n_episodes))]
        results = self._dispatch(jobs, "episode")
        return [EpisodeRecord.from_dict(results[k]) for k in range(int(n_episodes))]

    def validate(self) -> List[dict]:
        jobs = [(("validate", iid), iid) for iid in self.val_ids]
        results = self._dispatch(jobs, "val")
        return [dict(results[iid], instance_id=iid) for iid in self.val_ids]

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close",))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(5)
            if p.is_alive():
                p.terminate()
