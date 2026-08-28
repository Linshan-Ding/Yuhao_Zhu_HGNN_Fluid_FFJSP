"""剪枝量化 + oracle 保留率 + eps_f 敏感性（论文 Table T-NEW-4、Fig F-NEW-2）。

产出 result/pruning_stats.csv 与 result/pruning_sensitivity.csv。
"""
import time

import numpy as np
import torch

from _bootstrap import ROOT, checkpoints, done, step

from agent.networks import ActorCritic
from agent.ppo import PPOAgent
from analysis.pruning import analyse_instance
from configs.config import load_config
from data.dataset import read_index
from data.generator import load_instance_csv
from environment.env import SchedulingEnv
from result.logger import append_rows

ORACLE_ROLLOUTS = 8
ORACLE_EVERY = 20
EPS_GRID = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

STATS_COLUMNS = ["instance_id", "A_feas_mean", "A_feas_max", "A_f_mean", "A_f_max",
                 "prune_ratio", "p_singleton", "fallback_rate", "retention_all",
                 "retention_crit", "retention_se", "support_mean", "n_epochs",
                 "delta_eta", "t_lp_ms", "t_enc_ms", "t_pol_ms", "zeta"]
SENS_COLUMNS = ["eps_f", "prune_ratio", "retention_all", "retention_crit", "eta",
                "decision_time_ms"]

t0 = time.time()
cfg = load_config()
rng = np.random.default_rng()
ckpts = checkpoints("fshgrl_run*")
ckpt = ckpts[0] if ckpts else None
if ckpt is None:
    raise SystemExit("[FAIL] 未找到 fshgrl_run*/checkpoint_best.pt，请先运行 run_02")

net = ActorCritic(cfg)
net.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
net.eval()
agent = PPOAgent(net, cfg, torch.device("cpu"))


def policy(env, actions, sol):
    return agent.act(env.observation(actions, sol), env.problem.n_stage, 0.0, greedy=True)[0]


def run_episode(env, chooser):
    started = time.perf_counter()
    while not env.done:
        actions, sol = env.candidate_actions()
        if not actions:
            break
        if env.step(actions[chooser(env, actions, sol)])[1]:
            break
    return env, time.perf_counter() - started


step("逐算例剪枝统计与 oracle 保留率（main 档）")
rows = []
for meta in read_index("main"):
    inst = load_instance_csv(meta["path"], meta["tier"], meta["instance_id"])
    record = analyse_instance(lambda: SchedulingEnv(inst, cfg), meta["instance_id"], policy,
                              oracle_rollouts=ORACLE_ROLLOUTS, oracle_every=ORACLE_EVERY)
    summary = record.summary()

    # delta_eta：与不剪枝变体（FSHGRL-NoFP）在同一算例上的差
    env_pruned, _ = run_episode(SchedulingEnv(inst, cfg), policy)
    cfg_nofp = load_config(["ablation/nofp.yaml"])
    env_full, _ = run_episode(SchedulingEnv(inst, cfg_nofp), policy)
    summary["delta_eta"] = round(env_full.eta - env_pruned.eta, 6)

    stats = env_pruned.fluid.stats
    steps = max(env_pruned.step_count, 1)
    summary["t_lp_ms"] = round(1000.0 * stats.solve_seconds / steps, 4)
    summary["t_enc_ms"] = round(1000.0 * env_pruned.stats.t_obs / steps, 4)
    summary["t_pol_ms"] = ""
    summary["zeta"] = round(stats.zeta, 4)
    rows.append(summary)
    print(f"  {meta['instance_id']}: prune={summary['prune_ratio']}% "
          f"retention={summary['retention_all']}% P(|A_f|=1)={summary['p_singleton']}", flush=True)
append_rows(ROOT / "result" / "pruning_stats.csv", rows, STATS_COLUMNS)

step("eps_f 敏感性扫描")
sens = []
subset = read_index("main")[:5]
for eps in EPS_GRID:
    cfg_eps = load_config()
    cfg_eps.set("action_space.eps_f", float(eps))
    ratios, retentions, retentions_c, etas, times = [], [], [], [], []
    for meta in subset:
        inst = load_instance_csv(meta["path"], meta["tier"], meta["instance_id"])
        record = analyse_instance(lambda: SchedulingEnv(inst, cfg_eps), meta["instance_id"],
                                  policy, oracle_rollouts=4, oracle_every=40)
        s = record.summary()
        ratios.append(s["prune_ratio"])
        retentions.append(s["retention_all"])
        if np.isfinite(s["retention_crit"]):
            retentions_c.append(s["retention_crit"])
        env, elapsed = run_episode(SchedulingEnv(inst, cfg_eps), policy)
        etas.append(env.eta)
        times.append(1000.0 * elapsed / max(env.step_count, 1))
    sens.append({"eps_f": eps, "prune_ratio": round(float(np.mean(ratios)), 2),
                 "retention_all": round(float(np.mean(retentions)), 2),
                 "retention_crit": round(float(np.mean(retentions_c)), 2) if retentions_c else "",
                 "eta": round(float(np.mean(etas)), 4),
                 "decision_time_ms": round(float(np.mean(times)), 4)})
    print(f"  eps_f={eps:g}: prune={sens[-1]['prune_ratio']}% "
          f"retention={sens[-1]['retention_all']}% eta={sens[-1]['eta']}", flush=True)
append_rows(ROOT / "result" / "pruning_sensitivity.csv", sens, SENS_COLUMNS)

done(t0, ROOT / "result" / "pruning_stats.csv", ROOT / "result" / "pruning_sensitivity.csv")
