"""到达强度扫描、到达过程对照与分布外泛化（论文 Table T-NEW-8、Fig F-NEW-4）。

三种到达过程按构造共享同一平均到达率，因此它们之间的差异只反映突发性、不混入负荷差异。
OOD 档**不重训**——之所以可行，是因为图规模由 |O| x |M| 固定、剪枝后动作空间的界与订单数无关。
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
from result.logger import append_rows

N_ROLLOUT = 5
ARRIVAL_COLUMNS = ["instance_id", "E_dt", "rho_sys", "iota", "regime", "arrival_process",
                   "eta", "nu", "phi_star_mean", "decision_time_ms"]
OOD_COLUMNS = ["condition", "method", "eta", "eta_matched", "retention"]
SHIFT_COLUMNS = ["shift_axis", "train_cond", "test_cond", "eta", "retention"]

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


def ours(env, actions, sol):
    return agent.act(env.observation(actions, sol), env.problem.n_stage, 0.0, greedy=True)[0]


def play(inst, chooser):
    env = SchedulingEnv(inst, cfg)
    started = time.perf_counter()
    while not env.done:
        actions, sol = env.candidate_actions()
        if not actions:
            break
        if env.step(actions[chooser(env, actions, sol)])[1]:
            break
    ms = 1000.0 * (time.perf_counter() - started) / max(env.step_count, 1)
    phi = float(np.mean(env.stats.phi_star)) if env.stats.phi_star else float("nan")
    return env.eta, env.nu, phi, ms


step("到达强度扫描 x 到达过程（arrival 档，跨饱和点）")
rows = []
for meta in read_index("arrival"):
    inst = load_instance_csv(meta["path"], meta["tier"], meta["instance_id"])
    results = [play(inst, ours) for _ in range(N_ROLLOUT)]
    rows.append({"instance_id": meta["instance_id"], "E_dt": meta["E_dt"],
                 "rho_sys": meta["rho_sys"], "iota": meta["iota"], "regime": meta["regime"],
                 "arrival_process": meta["arrival_process"],
                 "eta": round(float(np.mean([r[0] for r in results])), 5),
                 "nu": round(float(np.mean([r[1] for r in results])), 5),
                 "phi_star_mean": round(float(np.nanmean([r[2] for r in results])), 4),
                 "decision_time_ms": round(float(np.mean([r[3] for r in results])), 4)})
    print(f"  {meta['instance_id']}: rho={meta['rho_sys']} eta={rows[-1]['eta']} "
          f"nu={rows[-1]['nu']}", flush=True)
append_rows(ROOT / "result" / "arrival_results.csv", rows, ARRIVAL_COLUMNS)

step("分布外泛化（ood 档，不重训）")
matched = float(np.mean([r["eta"] for r in rows if r["regime"] != "overloaded"]) or 0.0)
ood_rows = []
for meta in read_index("ood"):
    inst = load_instance_csv(meta["path"], meta["tier"], meta["instance_id"])
    eta_ours = float(np.mean([play(inst, ours)[0] for _ in range(N_ROLLOUT)]))
    # MOR/FIFO/MWKR/SPT/EDD 在给定算例上是确定性的，重复 rollout 不带来信息；
    # 只有 Random 与 RRC 有随机性，故只对这两条做多次 rollout。
    eta_pdr = max(
        float(np.mean([play(inst, lambda e, a, s, r=rule: rule_select(r, e, a, rng))[0]
                       for _ in range(N_ROLLOUT if rule in ("Random", "RRC") else 1)]))
        for rule in RULES)
    for method, value in (("FSHGRL", eta_ours), ("Best PDR", eta_pdr)):
        ood_rows.append({"condition": meta["instance_id"].replace("ood_", ""), "method": method,
                         "eta": round(value, 5), "eta_matched": round(matched, 5),
                         "retention": round(value / matched, 4) if matched > 0 else ""})
    print(f"  {meta['instance_id']}: FSHGRL={eta_ours:.4f} bestPDR={eta_pdr:.4f}", flush=True)
append_rows(ROOT / "result" / "ood_results.csv", ood_rows, OOD_COLUMNS)

step("分布偏移矩阵（到达强度轴）")
shift_rows = []
by_gap = {}
for r in rows:
    by_gap.setdefault(r["E_dt"], []).append(r["eta"])
conditions = sorted(by_gap, key=lambda v: float(v))
for train_cond in conditions:
    base = float(np.mean(by_gap[train_cond]))
    for test_cond in conditions:
        value = float(np.mean(by_gap[test_cond]))
        shift_rows.append({"shift_axis": "arrival_intensity", "train_cond": train_cond,
                           "test_cond": test_cond, "eta": round(value, 5),
                           "retention": round(value / base, 4) if base > 0 else ""})
append_rows(ROOT / "result" / "shift_matrix.csv", shift_rows, SHIFT_COLUMNS)

done(t0, ROOT / "result" / "arrival_results.csv", ROOT / "result" / "ood_results.csv",
     ROOT / "result" / "shift_matrix.csv")
