"""小档精确求解、解回放校验与最优性间隙（论文 Table T-NEW-5、占位符 A3 / P-MILPCHK）。

CP-SAT 为默认路径（免授权）；有 Gurobi 授权时同时求解 MILP，两个求解器一致本身
就是对公式化的一次独立检查。
"""
import time

import numpy as np
import torch

from _bootstrap import ROOT, checkpoints, done, step

from agent.baselines.rules import RULES, select as rule_select
from agent.networks import ActorCritic
from agent.ppo import PPOAgent
from configs.config import load_config
from data.dataset import read_index
from data.generator import load_instance_csv
from environment.env import SchedulingEnv
from environment.problem import Problem
from exact.milp import replay_check, solve_cpsat, solve_gurobi
from result.logger import append_rows

TIME_LIMIT_S = 3600.0
COLUMNS = ["instance_id", "S", "DDT", "eta_off_cpsat", "eta_off_gurobi", "solver_status",
           "solver_time_s", "replay_match", "eta_fshgrl", "eta_best_pdr", "eta_best_drl",
           "abs_gap", "rel_gap"]

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


def play(env, chooser):
    while not env.done:
        actions, sol = env.candidate_actions()
        if not actions:
            break
        if env.step(actions[chooser(env, actions, sol)])[1]:
            break
    return env.eta


step("small 档精确求解 + 回放校验 + 最优性间隙")
rows, matches = [], 0
for meta in read_index("small"):
    inst = load_instance_csv(meta["path"], meta["tier"], meta["instance_id"])
    problem = Problem(inst)
    cp = solve_cpsat(problem, time_limit_s=TIME_LIMIT_S)
    check = replay_check(problem, cp)
    matches += int(check["match"])
    gb = solve_gurobi(problem, time_limit_s=TIME_LIMIT_S)

    eta_ours = play(SchedulingEnv(inst, cfg),
                    lambda e, a, s: agent.act(e.observation(a, s), e.problem.n_stage,
                                              0.0, greedy=True)[0])
    eta_pdr = max(play(SchedulingEnv(inst, cfg),
                       lambda e, a, s, r=rule: rule_select(r, e, a, rng)) for rule in RULES)
    rows.append({
        "instance_id": meta["instance_id"], "S": meta["S"], "DDT": meta["DDT"],
        "eta_off_cpsat": round(cp.eta, 6),
        "eta_off_gurobi": round(gb.eta, 6) if np.isfinite(gb.eta) else "",
        "solver_status": cp.status, "solver_time_s": round(cp.seconds, 3),
        "replay_match": int(check["match"]), "eta_fshgrl": round(eta_ours, 6),
        "eta_best_pdr": round(eta_pdr, 6), "eta_best_drl": "",
        "abs_gap": round(cp.eta - eta_ours, 6),
        "rel_gap": round(100.0 * (cp.eta - eta_ours) / cp.eta, 3) if cp.eta > 0 else "",
    })
    print(f"  {meta['instance_id']}: eta_off={cp.eta:.4f} eta_ours={eta_ours:.4f} "
          f"replay={check['match']}", flush=True)

append_rows(ROOT / "result" / "exact_results.csv", rows, COLUMNS)
print(f"\n[P-MILPCHK] 解回放一致的算例：{matches}/{len(rows)}", flush=True)
done(t0, ROOT / "result" / "exact_results.csv")
