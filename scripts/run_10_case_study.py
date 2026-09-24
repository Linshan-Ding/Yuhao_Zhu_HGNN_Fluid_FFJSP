"""构造的 3D 打印情景（论文案例研究表，占位符 C-DDT600 / C-DDT900 / C-DDT1200）。

九个算例 = 3 个订单规模 x 3 个交期水平，6 阶段工艺链、阶段内异构设备。
这是情景可迁移性评估，不是工业验证——算例参数来自公开工艺数据与专家估计，
不来自车间执行日志，论文中已按此措辞。

FSHGRL 的每个 run 在每个算例上做一次贪心 rollout（策略在固定算例上是确定性的，重复
rollout 只会把同一个数抄几遍）：eta_best 取各 run 的最大值，eta_avg 取均值，置信区间是
跨 run 的 BCa，与表注"over R3 runs"一致。规则中 Random / RRC 取多次 rollout 的均值，
其余规则一次；best DRL 是三个学习基线各自全部 run 的均值中的最大者。
每次运行先清掉旧的 case3d_results.csv，误重跑不会追加重复行。
"""
import time

import numpy as np

from _bootstrap import ROOT, checkpoints, done, step

from agent.baselines.rules import RULES, STOCHASTIC_RULES
from analysis.stats import bca_ci
from configs.config import load_config
from data.dataset import read_index
from data.generator import load_instance_csv
from environment.env import SchedulingEnv
from environment.problem import Problem
from eval import _run_episode, make_chooser
from result.logger import append_rows

OUT = ROOT / "result" / "case3d_results.csv"
COLUMNS = ["case", "DDT", "S", "infeasible_share", "eta_best", "eta_avg", "ci_lo", "ci_hi", "n_runs",
           "decision_time_s", "eta_best_rule", "best_rule", "eta_avg_rule", "eta_best_drl", "best_drl",
           "imp_pct", "gap_pct"]
BASELINE_TAGS = {"drlg": "DRLG", "ahpdqn": "AHP-DQN", "hsddqn": "HSDDQN"}


def infeasible_share(inst):
    """到达时就已不可能按时完成的订单比例（交期宽松度 < 最短剩余路径）。

    这部分订单任何方法、包括离线最优都救不回来；比例过高时整张表只剩噪声。
    """
    problem = Problem(inst)
    return float(np.mean([inst.due_dates[s] - inst.arrival_times[s] < problem.residual_from(s, 0)
                          for s in range(problem.n_order)]))


def play(inst, cfg, chooser):
    row = _run_episode(SchedulingEnv(inst, cfg), chooser)
    return float(row["eta"]), float(row["decision_time_ms"]) * int(row["steps"]) / 1000.0


def main():
    t0 = time.time()
    cfg = load_config()
    rng = np.random.default_rng()
    rollouts = int(cfg.get("runtime.eval_rollouts", 10))
    ours = checkpoints("fshgrl_run*")
    if not ours:
        raise SystemExit("[FAIL] 未找到 fshgrl_run*/checkpoint_best.pt，请先运行 run_02")
    policies = [make_chooser("FSHGRL", cfg, c, rng) for c in ours]
    rules = {rule: make_chooser(rule, cfg, None, rng) for rule in RULES}
    drl = {name: [make_chooser(name, cfg, c, rng) for c in checkpoints(f"{tag}_run*")]
           for tag, name in BASELINE_TAGS.items()}
    drl = {name: choosers for name, choosers in drl.items() if choosers}
    if OUT.exists():
        OUT.unlink()
        print(f"[INFO] 已清空旧的 {OUT.name}，结果将重新生成", flush=True)

    step(f"九个 3D 打印情景算例（FSHGRL {len(ours)} 个 run，各一次贪心 rollout）")
    rows = []
    for idx, meta in enumerate(sorted(read_index("case3d"),
                                      key=lambda r: (float(r["DDT"]), int(r["S"]))), start=1):
        inst = load_instance_csv(meta["path"], meta["tier"], meta["instance_id"])
        share = infeasible_share(inst)
        if share >= 0.5:
            print(f"[WARN] {meta['instance_id']}：{share:.0%} 的订单到达时已不可能按时完成，"
                  f"该算例的 eta 对任何方法都接近 0（见 README §0.4）", flush=True)
        runs = [play(inst, cfg, chooser) for chooser in policies]
        etas = [eta for eta, _ in runs]
        lo, hi = bca_ci(etas, n_boot=10000)

        rule_eta = {rule: float(np.mean([play(inst, cfg, chooser)[0] for _ in
                                         range(rollouts if rule in STOCHASTIC_RULES else 1)]))
                    for rule, chooser in rules.items()}
        best_rule = max(rule_eta, key=rule_eta.get)
        avg_rule = float(np.mean(list(rule_eta.values())))
        drl_eta = {name: float(np.mean([play(inst, cfg, chooser)[0] for chooser in choosers]))
                   for name, choosers in drl.items()}
        best_drl = max(drl_eta, key=drl_eta.get) if drl_eta else ""

        avg = float(np.mean(etas))
        best = rule_eta[best_rule]
        rows.append({"case": f"C{idx}", "DDT": meta["DDT"], "S": meta["S"],
                     "infeasible_share": round(share, 4),
                     "eta_best": round(max(etas), 4), "eta_avg": round(avg, 4),
                     "ci_lo": round(lo, 4) if np.isfinite(lo) else "",
                     "ci_hi": round(hi, 4) if np.isfinite(hi) else "", "n_runs": len(etas),
                     "decision_time_s": round(float(np.mean([s for _, s in runs])), 4),
                     "eta_best_rule": round(best, 4), "best_rule": best_rule,
                     "eta_avg_rule": round(avg_rule, 4),
                     "eta_best_drl": round(drl_eta[best_drl], 4) if best_drl else "", "best_drl": best_drl,
                     "imp_pct": round(100.0 * (avg - avg_rule) / avg_rule, 2) if avg_rule > 0 else "",
                     "gap_pct": round(100.0 * (avg - best) / avg, 2) if avg > 0 else ""})
        print(f"  C{idx} DDT={meta['DDT']} S={meta['S']}: eta_avg={avg:.4f} [{lo:.4f}, {hi:.4f}] "
              f"best_rule={best:.4f} ({best_rule})", flush=True)

    append_rows(OUT, rows, COLUMNS)
    done(t0, OUT)


if __name__ == "__main__":
    main()
