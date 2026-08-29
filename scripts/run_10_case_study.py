"""构造的 3D 打印情景（论文案例研究表）。

九个算例 = 3 个订单规模 x 3 个交期水平，6 阶段工艺链、阶段内异构设备。
这是情景可迁移性评估，不是工业验证——算例参数来自公开工艺数据与专家估计，
不来自车间执行日志，论文中已按此措辞。
"""
import time

import numpy as np
import torch

from _bootstrap import ROOT, checkpoints, done, step

from agent.baselines.drl_baselines import BASELINES
from agent.baselines.rules import RULES, select as rule_select
from agent.networks import ActorCritic
from agent.ppo import PPOAgent
from analysis.stats import bca_ci
from configs.config import load_config
from data.dataset import read_index
from data.generator import load_instance_csv
from environment.env import SchedulingEnv
from result.logger import append_rows

N_ROLLOUT = 10
COLUMNS = ["case", "DDT", "S", "eta_best", "eta_avg", "ci_lo", "ci_hi", "decision_time_s",
           "eta_best_rule", "eta_avg_rule", "eta_best_drl", "imp_pct", "gap_pct"]

t0 = time.time()
cfg = load_config()
rng = np.random.default_rng()
ckpts = checkpoints("fshgrl_run*")
if not ckpts:
    raise SystemExit("[FAIL] 未找到 fshgrl_run*/checkpoint_best.pt，请先运行 run_02")
net = ActorCritic(cfg)
net.load_state_dict(torch.load(ckpts[0], map_location="cpu")["model"])
net.eval()
agent = PPOAgent(net, cfg, torch.device("cpu"))

drl_nets = {}
for tag, name in (("drlg", "DRLG"), ("ahpdqn", "AHP-DQN"), ("hsddqn", "HSDDQN")):
    found = checkpoints(f"{tag}_run*")
    if found:
        model = BASELINES[name](cfg)
        model.load_state_dict(torch.load(found[0], map_location="cpu")["model"], strict=False)
        model.eval()
        drl_nets[name] = model


def play(inst, chooser):
    env = SchedulingEnv(inst, cfg)
    started = time.perf_counter()
    while not env.done:
        actions, sol = env.candidate_actions()
        if not actions:
            break
        if env.step(actions[chooser(env, actions, sol)])[1]:
            break
    return env.eta, time.perf_counter() - started


step("九个 3D 打印情景算例")
rows = []
for idx, meta in enumerate(sorted(read_index("case3d"),
                                  key=lambda r: (float(r["DDT"]), int(r["S"]))), start=1):
    inst = load_instance_csv(meta["path"], meta["tier"], meta["instance_id"])
    ours = [play(inst, lambda e, a, s: agent.act(e.observation(a, s), e.problem.n_stage,
                                                 0.0, greedy=True)[0]) for _ in range(N_ROLLOUT)]
    etas = [r[0] for r in ours]
    seconds = float(np.mean([r[1] for r in ours]))
    lo, hi = bca_ci(etas, n_boot=10000)

    rule_means = {rule: float(np.mean([play(inst, lambda e, a, s, r=rule:
                                            rule_select(r, e, a, rng))[0]
                                       for _ in range(N_ROLLOUT)])) for rule in RULES}
    best_rule = max(rule_means.values())
    avg_rule = float(np.mean(list(rule_means.values())))
    best_drl = max((float(np.mean([play(inst, lambda e, a, s, m=model:
                                       m.act(e, a, e.observation(a, s), rng, greedy=True)[0])[0]
                                   for _ in range(N_ROLLOUT)]))
                    for model in drl_nets.values()), default=float("nan"))

    avg = float(np.mean(etas))
    rows.append({"case": f"C{idx}", "DDT": meta["DDT"], "S": meta["S"],
                 "eta_best": round(max(etas), 4), "eta_avg": round(avg, 4),
                 "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                 "decision_time_s": round(seconds, 4),
                 "eta_best_rule": round(best_rule, 4), "eta_avg_rule": round(avg_rule, 4),
                 "eta_best_drl": round(best_drl, 4) if np.isfinite(best_drl) else "",
                 "imp_pct": round(100.0 * (avg - avg_rule) / avg_rule, 2) if avg_rule > 0 else "",
                 "gap_pct": round(100.0 * (avg - best_rule) / avg, 2) if avg > 0 else ""})
    print(f"  C{idx} DDT={meta['DDT']} S={meta['S']}: eta_avg={avg:.4f} "
          f"[{lo:.4f}, {hi:.4f}] best_rule={best_rule:.4f}", flush=True)

append_rows(ROOT / "result" / "case3d_results.csv", rows, COLUMNS)
done(t0, ROOT / "result" / "case3d_results.csv")
