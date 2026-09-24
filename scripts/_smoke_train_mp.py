"""多进程训练链路冒烟（run_00 第 4 步）。任何一项不过即报错退出。

1 填充不变性：单样本前向与填充后的批量前向一致，填充位 log p 恰为 -inf；
2 归一化：联合分布每行和为 1，no-op 行概率 = sigmoid(门控 logit)，熵有限非负；
3 worker 往返（强制 spawn，2 个 worker）：标签、掩码、Σr = η、no-op 在末行；
4 PPO 更新：主进程重算的 log p 与 worker 记录一致，更新统计有限，有标签进损失；
5 可复现：同种子两次采集、1 与 2 个 worker 采到同一批记录；
6 贪心确定性：同一验证算例两次结果相同；
7 spawn 兼容：worker 入口与参数可 pickle；
8 三个学习基线各采 1 个 epoch 并完成一次更新；
9 checkpoint 保存后能重新加载并贪心跑通一条 episode。
"""
import pickle
import sys
import time

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401

from agent.batch import NEG, collate_records, single_batch, to_batch
from agent.buffer import EpisodeRecord
from agent.dqn import DQNLearner, HierDQNLearner
from agent.networks import ActorCritic, greedy_actions
from agent.ppo import PPOLearner
from agent.workers import CoHPolicy, WorkerPool, run_episode, worker_main
from configs.config import load_config
from data.dataset import read_index
from data.generator import load_instance_csv
from environment.env import SchedulingEnv, is_noop

torch.set_num_threads(1)
CPU = torch.device("cpu")


def fail(msg):
    raise SystemExit(f"[FAIL] {msg}")


def tiny_cfg(method="coh"):
    cfg = load_config(["coh.yaml"])
    cfg.set("training.method", method)
    cfg.set("param_table.order_count_range", [10, 15])
    cfg.set("rollout.episodes_per_epoch", 4)
    cfg.set("training.total_steps", 2000)
    cfg.set("dqn.learning_starts", 0)
    cfg.set("dqn.minibatch_size", 16)
    return cfg


def load(tier, iid):
    m = [r for r in read_index(tier) if r["instance_id"] == iid][0]
    return load_instance_csv(m["path"], m["tier"], m["instance_id"])


def observations(cfg):
    """三个候选数不同的 grid 观测 + 一个结构不同的 ood 观测。"""
    obs = []
    env = SchedulingEnv(load("grid", "grid_rho1p6_DDT1100_S100"), cfg)
    rng = np.random.default_rng(0)
    seen = set()
    while not env.done and len(obs) < 3:
        actions = env.candidate_actions()
        if not actions:
            break
        if len(actions) not in seen:
            seen.add(len(actions))
            obs.append(env.observation(actions))
        env.step(actions[rng.integers(len(actions))])
    env2 = SchedulingEnv(load("ood", "ood_R8J7"), cfg)
    actions = env2.candidate_actions()
    obs.append(env2.observation(actions))
    return obs


def padded_batch(obs_list, extra_rows=7):
    """把观测填充成一个批，候选轴多填 extra_rows 行、节点轴填到最大。"""
    steps = [{"obs": o, "action": 0, "env_action": 0, "logp": 0.0, "value": 0.0, "reward": 0.0,
              "done": True, "order": -1, "hold_id": -1, "extra": {}} for o in obs_list]
    recs = []
    for s in steps:
        class _E:  # 只为 from_steps 提供结束时的字段
            order_outcome = np.zeros(1, np.int8)
            def hold_labels(self): return {}
            eta = nu = held_share = 0.0
            step_count = 1
            class stats: noop_offered = noop_used = 0; n_candidates = [1]
            class problem: n_order = 1
            class inst: meta = {}
        recs.append(EpisodeRecord.from_steps([s], _E()))
    arrays = collate_records(recs)
    A = arrays["act_feat"].shape[1] + extra_rows
    for key in ("act_feat", "act_mask", "act_index"):
        arr = arrays[key]
        pad = np.zeros((arr.shape[0], A) + arr.shape[2:], arr.dtype)
        pad[:, :arr.shape[1]] = arr
        arrays[key] = pad
    return to_batch(arrays, CPU)


def check_padding_and_normalisation(cfg):
    torch.manual_seed(0)
    net = ActorCritic(cfg).eval()
    obs = observations(cfg)
    with torch.no_grad():
        singles = [net(single_batch(o)) for o in obs]
        batch = padded_batch(obs)
        out = net(batch)
    for i, (o, s) in enumerate(zip(obs, singles)):
        n = o["act_feat"].shape[0]
        if not torch.allclose(out.logp[i, :n], s.logp[0], atol=1e-5):
            fail(f"填充后 log p 不一致（观测 {i}）")
        if not torch.allclose(out.value[i], s.value[0], atol=1e-5):
            fail("填充后 value 不一致")
        if not torch.allclose(out.commit_logit[i, :n], s.commit_logit[0], atol=1e-5):
            fail("填充后 commit logit 不一致")
        if not torch.allclose(out.gate_logit[i], s.gate_logit[0], atol=1e-5) \
                or not torch.allclose(out.hold_logit[i], s.hold_logit[0], atol=1e-5):
            fail("填充后门控 / 等待前景 logit 不一致")
        if not torch.isneginf(out.logp[i, n:]).all():
            fail("填充位 log p 不是 -inf")
        total = float(out.logp[i, :n].exp().sum())
        if abs(total - 1.0) > 1e-5:
            fail(f"联合分布未归一：{total}")
        if is_noop_row(o):
            p_noop = float(out.logp[i, n - 1].exp())
            if abs(p_noop - float(torch.sigmoid(out.gate_logit[i]))) > 1e-6:
                fail("no-op 行概率 != sigmoid(gate logit)")
        if not (torch.isfinite(out.entropy[i]) and out.entropy[i] >= 0):
            fail("熵非有限或为负")
    print(f"  1-2 填充不变性与归一化：{len(obs)} 个观测（候选数 {[o['act_feat'].shape[0] for o in obs]}）通过", flush=True)


def is_noop_row(o):
    return bool(o["act_feat"][-1, -1] > 0.5)


def check_records(records, name):
    for rec in records:
        noop_steps = rec.env_action == (rec.act_mask.sum(1) - 1)
        noop_rows = rec.act_feat[np.arange(len(rec)), rec.env_action, -1] > 0.5
        if not np.array_equal(noop_steps & noop_rows, noop_rows):
            fail(f"{name}: no-op 不在末行")
        if not set(rec.commit_label[~noop_rows].tolist()) <= {0, 1}:
            fail(f"{name}: 派工步的承诺标签应为 0/1")
        if not (rec.commit_label[noop_rows] == -1).all():
            fail(f"{name}: no-op 步不应有承诺标签")
        if not set(rec.hold_label.tolist()) <= {-1, 0, 1} or not (rec.hold_label[~noop_rows] == -1).all():
            fail(f"{name}: 等待标签异常")
        if not rec.act_mask.any(1).all():
            fail(f"{name}: 有转移没有有效候选")
        if abs(rec.reward.sum() - rec.info["eta"]) > 1e-6:
            fail(f"{name}: Σr != η")
        if not 0.0 <= rec.info["held_share"] <= 1.0:
            fail(f"{name}: held_share 越界")


def main():
    cfg = tiny_cfg()
    check_padding_and_normalisation(cfg)
    val_meta = read_index("val")[:2]

    # 7 spawn 兼容
    pickle.dumps((worker_main, (0, cfg.to_dict(), "coh", None, 1, val_meta)))
    print("  7 worker 入口与参数可 pickle", flush=True)

    # 3 worker 往返
    learner = PPOLearner(ActorCritic(cfg), cfg, CPU, 2000)
    learner.seed(1)
    pool = WorkerPool(2, cfg.to_dict(), "coh", 1, val_meta)
    t0 = time.perf_counter()
    pool.broadcast(learner.policy_state_numpy())
    records = pool.collect(0, 4, {})
    dt = time.perf_counter() - t0
    if len(records) != 4:
        fail("采集的 episode 数不对")
    check_records(records, "coh")
    n_steps = sum(len(r) for r in records)
    print(f"  3 worker 往返：4 条 episode 共 {n_steps} 步，{dt:.1f}s（{n_steps / dt:.0f} 步/s，2 worker，含启动）", flush=True)

    # 4 更新前重算 log p 与 worker 一致；更新统计有限
    with torch.no_grad():
        arrays = collate_records(records)
        out = learner.net(to_batch(arrays, CPU))
        action = torch.from_numpy(np.concatenate([r.action for r in records]))
        logp_main = out.logp.gather(1, action.unsqueeze(1)).squeeze(1).numpy()
        logp_worker = np.concatenate([r.logp for r in records])
    if np.abs(logp_main - logp_worker).max() > 1e-4:
        fail(f"主进程与 worker 的 log p 不一致：{np.abs(logp_main - logp_worker).max()}")
    stats = learner.update(records, n_steps)
    for k, v in stats.items():
        if isinstance(v, float) and not np.isfinite(v):
            fail(f"更新统计 {k} 非有限")
    if stats.get("n_labels", 0) <= 0 or "commit_bce" not in stats:
        fail("辅助损失没有拿到标签")
    print(f"  4 PPO 更新：kl={stats['approx_kl']:.4f} commit_bce={stats['commit_bce']:.3f} "
          f"hold_bce={stats.get('hold_bce', float('nan')):.3f} n_labels={int(stats['n_labels'])}", flush=True)

    # 5 可复现
    pool.broadcast(learner.policy_state_numpy())
    again = pool.collect(3, 4, {})
    twice = pool.collect(3, 4, {})
    pool1 = WorkerPool(1, cfg.to_dict(), "coh", 1, val_meta)
    pool1.broadcast(learner.policy_state_numpy())
    once = pool1.collect(3, 4, {})
    for a, b, c in zip(again, twice, once):
        if not (np.array_equal(a.action, b.action) and np.array_equal(a.action, c.action)
                and a.info["eta"] == b.info["eta"] == c.info["eta"]):
            fail("同种子采集不可复现（或依赖 worker 数）")
    pool1.close()
    print("  5 同种子两次采集与 1/2 个 worker 采集逐位相同", flush=True)

    # 6 贪心确定性
    v1, v2 = pool.validate(), pool.validate()
    if any(a["eta"] != b["eta"] or a["steps"] != b["steps"] for a, b in zip(v1, v2)):
        fail("贪心验证不确定")
    print(f"  6 验证确定：eta={[round(v['eta'], 3) for v in v1]}", flush=True)
    pool.close()

    # 8 三个学习基线
    for method, cls in (("rule_ppo", None), ("dqn", DQNLearner), ("hdqn", HierDQNLearner)):
        mcfg = tiny_cfg(method)
        if method == "rule_ppo":
            from agent.baselines.rule_ppo import RulePolicyNet
            lrn = PPOLearner(RulePolicyNet(mcfg), mcfg, CPU, 2000)
        elif method == "dqn":
            from agent.baselines.dqn_nets import TripletQNet
            lrn = cls(TripletQNet(mcfg), mcfg, CPU, 2000)
        else:
            from agent.baselines.dqn_nets import HierQNet
            lrn = cls(HierQNet(mcfg), mcfg, CPU, 2000)
        lrn.seed(1)
        bpool = WorkerPool(2, mcfg.to_dict(), method, 2, val_meta)
        bpool.broadcast(lrn.policy_state_numpy())
        recs = bpool.collect(0, 4, lrn.behaviour_params(0))
        check_records(recs, method)
        st = lrn.update(recs, sum(len(r) for r in recs))
        if any(isinstance(v, float) and not np.isfinite(v) for v in st.values()):
            fail(f"{method} 更新统计非有限")
        val = bpool.validate()
        bpool.close()
        key = "q_loss" if method != "rule_ppo" else "policy_loss"
        print(f"  8 {method}: {sum(len(r) for r in recs)} 步，{key}={st.get(key, float('nan')):.4f}，"
              f"验证 eta={[round(v['eta'], 3) for v in val]}", flush=True)

    # 9 checkpoint 往返
    path = ROOT / "result" / "smoke_ckpt.pt"
    torch.save({"model": learner.net.state_dict(), "method": "coh"}, path)
    net = ActorCritic(cfg)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    policy = CoHPolicy(cfg)
    policy.net.load_state_dict(net.state_dict())
    env = SchedulingEnv(load("val", val_meta[0]["instance_id"]), cfg)
    run_episode(env, policy, None, {}, greedy=True, record=False)
    path.unlink()
    print(f"  9 checkpoint 重载后贪心 episode：eta={env.eta:.3f} held={env.held_share:.3f}", flush=True)
    print("  多进程训练链路冒烟通过", flush=True)


if __name__ == "__main__":
    main()
