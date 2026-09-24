"""CoH 冒烟（run_00 第 5 步）。

(1) 新旧前向一致：CoH 开关全关时 `log_policy` 与 `forward` 的 softmax 逐位一致、贪心动作序列
    相同——先用随机初始化网络，再用 result/fshgrl_run1/checkpoint_best.pt（存在时）。
    这保证旧 checkpoint 的评测结果在加入 CoH 之后不变。
(2) p2 配置（承诺评估器 + 门控 + 等待前景评估器）用 smoke_coh_run1 的 checkpoint 走一条按策略
    采样的 episode：联合分布归一、标签回填完整（订单结果无 -1、等待标签只在 {-1,0,1}）、
    held_share 在 [0,1]；训练日志里 commit_bce 有限且 n_labels > 0。
"""
import csv
import math

import numpy as np
import torch

from _bootstrap import ROOT

from agent.networks import ActorCritic, obs_to_tensors
from configs.config import load_config
from data.dataset import read_index
from data.generator import load_instance_csv
from environment.env import SchedulingEnv

CPU = torch.device("cpu")


def instances(n):
    return [load_instance_csv(m["path"], m["tier"], m["instance_id"]) for m in read_index("main")[:n]]


def check_consistency(cfg, ckpt=None) -> int:
    net = ActorCritic(cfg)
    if ckpt is not None:
        net.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    net.eval()
    steps = 0
    with torch.no_grad():
        for inst in instances(2):
            env = SchedulingEnv(inst, cfg)
            while not env.done:
                actions, sol = env.candidate_actions()
                if not actions:
                    break
                obs = obs_to_tensors(env.observation(actions, sol), CPU)
                logits, v_old = net(obs, env.problem.n_stage)
                logp, v_new, _ = net.log_policy(obs, env.problem.n_stage)
                same = (torch.allclose(torch.log_softmax(logits, dim=-1), logp, atol=1e-6)
                        and int(torch.argmax(logits)) == int(torch.argmax(logp))
                        and torch.allclose(v_old, v_new))
                if not same:
                    raise SystemExit(f"[FAIL] forward 与 log_policy 不一致（第 {steps} 步）")
                steps += 1
                if env.step(actions[int(torch.argmax(logp))])[1]:
                    break
    return steps


def check_coh(cfg, ckpt) -> None:
    net = ActorCritic(cfg)
    net.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    net.eval()
    env = SchedulingEnv(instances(1)[0], cfg)
    n_hold = n_step = 0
    with torch.no_grad():
        while not env.done:
            actions, sol = env.candidate_actions()
            if not actions:
                break
            obs = obs_to_tensors(env.observation(actions, sol), CPU)
            logp, value, aux = net.log_policy(obs, env.problem.n_stage)
            total = float(logp.exp().sum())
            if abs(total - 1.0) > 1e-5 or not torch.isfinite(logp).all():
                raise SystemExit(f"[FAIL] 联合分布未归一：sum={total}")
            if aux["commit_logit"] is None or not torch.isfinite(aux["commit_logit"]).all():
                raise SystemExit("[FAIL] 承诺评估器无输出或非有限")
            idx = int(torch.multinomial(logp.exp(), 1).item())     # 按策略采样，覆盖等待分支
            _, done, info = env.step(actions[idx])
            n_hold += int(info.get("hold_id", -1) >= 0)
            n_step += 1
            if done:
                break
    labels = env.hold_labels()
    if set(labels.values()) - {-1, 0, 1} or len(labels) != n_hold:
        raise SystemExit(f"[FAIL] 等待标签异常：{len(labels)} 条 / {n_hold} 次等待")
    if env.done and (env.order_outcome < 0).any():
        raise SystemExit("[FAIL] episode 结束后仍有订单结果未定")
    if not 0.0 <= env.held_share <= 1.0:
        raise SystemExit(f"[FAIL] held_share 越界：{env.held_share}")
    dist = {v: sum(1 for x in labels.values() if x == v) for v in (-1, 0, 1)}
    print(f"  p2 episode：{n_step} 步，{n_hold} 次等待，等待标签 {dist}，"
          f"订单结果按时/未按时 = {int((env.order_outcome == 1).sum())}/"
          f"{int((env.order_outcome == 0).sum())}，held_share={env.held_share:.3f}，"
          f"eta={env.eta:.3f}", flush=True)


def check_log(run_dir) -> None:
    with (run_dir / "log.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    last = rows[-1]
    n_labels = float(last.get("n_labels") or 0)
    bce = float(last.get("commit_bce") or "nan")
    if n_labels <= 0 or not math.isfinite(bce):
        raise SystemExit(f"[FAIL] 训练日志：n_labels={n_labels} commit_bce={bce}")
    print(f"  训练日志：n_labels={int(n_labels)} commit_bce={bce:.4f} "
          f"commit_brier={last.get('commit_brier')} hold_bce={last.get('hold_bce')} "
          f"held_share={last.get('held_share')}", flush=True)


if __name__ == "__main__":
    base = load_config()
    print(f"  随机网络：forward 与 log_policy 在 {check_consistency(base)} 个决策点一致", flush=True)
    ckpt = ROOT / "result" / "fshgrl_run1" / "checkpoint_best.pt"
    if ckpt.exists():
        print(f"  fshgrl_run1：forward 与 log_policy 在 {check_consistency(base, ckpt)} 个决策点一致",
              flush=True)
    else:
        print("  [跳过] 没有 result/fshgrl_run1/checkpoint_best.pt，只做随机网络的一致性检查", flush=True)
    cfg = load_config(["coh/p2_gate.yaml"])
    run_dir = ROOT / "result" / "smoke_coh_run1"
    check_log(run_dir)
    check_coh(cfg, run_dir / "checkpoint_best.pt")
    print("  CoH 冒烟通过", flush=True)
