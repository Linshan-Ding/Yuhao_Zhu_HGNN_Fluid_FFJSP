"""小档精确求解、回放校验、在线重优化与最优性间隙（论文 Table T-NEW-5，占位符 A3 / S7 / S8 / P-MILPCHK）。

每个算例做六件事，全部不需要商业求解器授权：
  1. CP-SAT 离线最优：保序整数时间尺度，与连续时间模型逐解等价（见 exact/milp.py）；
  2. HiGHS MILP：稿件 Eqs. (1)-(11) 的直译独立求解（SciPy 自带），限时 --milp-time-limit；
  3. 证书：CP-SAT 的最优排程代入字面 MILP 的约束矩阵逐行核对；
  4. 回放：CP-SAT 与 MILP 的排程分别驱动离散事件环境，环境记下的按时完工数须与求解器一致；
  5. 在线滚动重优化：每个到达时刻用 CP-SAT 精确重排已知订单，排程同样回放；
  6. 贪心评测 FSHGRL 全部 run、全部规则（Random / RRC 取多次 rollout 均值）、三个学习基线全部 run。

eta_off 取证得最优的求解器的值。两个都证得最优时必须相等；只有一个证得最优时，它必须
落在另一个的 [可行解, 对偶上界] 之内。任何一条不成立都记为 CONFLICT。
逐算例落盘、可断点续跑：result/exact_results.csv 里已有的算例直接跳过，要从头重算先删该文件。
--jobs N 按算例并行（HiGHS 的分支定界基本是单线程，按算例并行最划算）。
"""
import argparse
import csv
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, checkpoints, done, step

from data.dataset import read_index
from result.logger import append_rows

OUT = ROOT / "result" / "exact_results.csv"
COLUMNS = ["instance_id", "S", "DDT",
           "eta_off", "eta_off_source",
           "eta_off_cpsat", "cpsat_status", "cpsat_time_s",
           "eta_off_milp", "milp_status", "milp_upper", "milp_time_s",
           "cert_cpsat_in_milp", "replay_match", "replay_match_milp",
           "eta_online", "online_solves", "online_all_optimal", "online_time_s", "replay_match_online",
           "eta_fshgrl", "eta_fshgrl_sd", "n_fshgrl_runs",
           "eta_best_pdr", "best_pdr", "eta_best_drl", "best_drl",
           "abs_gap", "rel_gap"]
BASELINE_TAGS = {"drlg": "DRLG", "ahpdqn": "AHP-DQN", "hsddqn": "HSDDQN"}


def _r(value, digits=6):
    return round(float(value), digits) if value is not None and np.isfinite(value) else ""


def _mean_eta(inst, cfg, method, ckpt, rollouts):
    from agent.baselines.rules import STOCHASTIC_RULES
    from environment.env import SchedulingEnv
    from eval import _run_episode, make_chooser

    rng = np.random.default_rng()
    chooser = make_chooser(method, cfg, Path(ckpt) if ckpt else None, rng)
    n = rollouts if method in STOCHASTIC_RULES else 1
    return float(np.mean([_run_episode(SchedulingEnv(inst, cfg), chooser)["eta"] for _ in range(n)]))


def _evaluate_instance(meta, opts):
    """一个算例的全部计算。放在模块顶层，spawn 启动的子进程才能导入它。"""
    import torch

    from agent.baselines.rules import RULES
    from configs.config import load_config
    from data.generator import load_instance_csv
    from environment.problem import Problem
    from exact.milp import (ExactResult, milp_certificate, replay_check, solve_cpsat, solve_milp,
                            solve_online_reoptimization)

    torch.set_num_threads(1)
    cfg = load_config()
    inst = load_instance_csv(meta["path"], meta["tier"], meta["instance_id"])
    problem = Problem(inst)
    solved = ("OPTIMAL", "FEASIBLE", "TIME_LIMIT")

    cp = solve_cpsat(problem, time_limit_s=opts["cpsat_time_limit"], workers=opts["cpsat_workers"])
    cp_ok = cp.status in solved
    replay = replay_check(problem, cp, cfg) if cp_ok else {"match": False}
    cert = milp_certificate(problem, cp) if cp_ok else {"ok": False}

    if opts["milp_time_limit"] > 0:
        ml = solve_milp(problem, time_limit_s=opts["milp_time_limit"])
    else:
        ml = ExactResult(eta=float("nan"), status="SKIPPED", solver="highs-milp", seconds=0.0, assignment={})
    ml_ok = ml.status in solved
    replay_ml = replay_check(problem, ml, cfg) if ml_ok else {"match": False}

    # 在线重优化固定单线程：提交哪份计划取决于求解轨迹，单线程才逐位可复现
    on = solve_online_reoptimization(problem, time_limit_per_solve=opts["online_time_limit"])
    replay_on = replay_check(problem, on, cfg)

    # 交叉核对：两个都证得最优时必须相等；只有一个证得最优时，它必须落在另一个的
    # [可行解, 证得的上界] 之内。任何一条不成立都记为 CONFLICT。
    def inside(value, other):
        return ((not np.isfinite(other.eta) or other.eta <= value + 1e-9)
                and (not np.isfinite(other.upper) or value <= other.upper + 1e-9))

    if cp.status == "OPTIMAL" and ml.status == "OPTIMAL":
        eta_off, source = cp.eta, ("cpsat=milp" if cp.n_completed == ml.n_completed else "CONFLICT")
    elif cp.status == "OPTIMAL":
        eta_off, source = cp.eta, ("cpsat" if inside(cp.eta, ml) else "CONFLICT")
    elif ml.status == "OPTIMAL":
        eta_off, source = ml.eta, ("milp" if inside(ml.eta, cp) else "CONFLICT")
    else:                                        # 两个都没证得最优：取较好的可行解，并如实标注
        eta_off = max((x.eta for x in (cp, ml) if np.isfinite(x.eta)), default=float("nan"))
        source = "incumbent"

    ours = [_mean_eta(inst, cfg, "FSHGRL", ckpt, 1) for ckpt in opts["fshgrl"]]
    rule_eta = {rule: _mean_eta(inst, cfg, rule, None, opts["rollouts"]) for rule in RULES}
    best_pdr = max(rule_eta, key=rule_eta.get)
    drl_eta = {name: float(np.mean([_mean_eta(inst, cfg, name, ckpt, 1) for ckpt in ckpts]))
               for name, ckpts in opts["drl"].items() if ckpts}
    best_drl = max(drl_eta, key=drl_eta.get) if drl_eta else ""

    eta_ours = float(np.mean(ours))
    return {
        "instance_id": meta["instance_id"], "S": meta["S"], "DDT": meta["DDT"],
        "eta_off": _r(eta_off), "eta_off_source": source,
        "eta_off_cpsat": _r(cp.eta), "cpsat_status": cp.status, "cpsat_time_s": _r(cp.seconds, 3),
        "eta_off_milp": _r(ml.eta), "milp_status": ml.status, "milp_upper": _r(ml.upper),
        "milp_time_s": _r(ml.seconds, 3),
        "cert_cpsat_in_milp": int(bool(cert["ok"])), "replay_match": int(bool(replay["match"])),
        "replay_match_milp": int(bool(replay_ml["match"])) if ml_ok else "",
        "eta_online": _r(on.eta), "online_solves": on.n_solves,
        "online_all_optimal": int(on.all_optimal), "online_time_s": _r(on.seconds, 3),
        "replay_match_online": int(bool(replay_on["match"])),
        "eta_fshgrl": _r(eta_ours), "eta_fshgrl_sd": _r(np.std(ours, ddof=1) if len(ours) > 1 else 0.0),
        "n_fshgrl_runs": len(ours),
        "eta_best_pdr": _r(rule_eta[best_pdr]), "best_pdr": best_pdr,
        "eta_best_drl": _r(drl_eta[best_drl]) if best_drl else "", "best_drl": best_drl,
        "abs_gap": _r(eta_off - eta_ours),
        "rel_gap": _r(100.0 * (eta_off - eta_ours) / eta_off, 3) if eta_off > 0 else "",
    }


def _read_rows():
    if not OUT.exists():
        return []
    with OUT.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=1, help="按算例并行的进程数")
    parser.add_argument("--cpsat-workers", type=int, default=0,
                        help="每个进程里 CP-SAT 的线程数；默认 CPU 核数 / jobs（至多 8）")
    parser.add_argument("--cpsat-time-limit", type=float, default=3600.0)
    parser.add_argument("--milp-time-limit", type=float, default=3600.0,
                        help="HiGHS MILP 每个算例的时限（秒），与论文一致；0 表示跳过 MILP")
    parser.add_argument("--online-time-limit", type=float, default=60.0,
                        help="在线重优化每次重解的时限（秒）")
    parser.add_argument("--instances", nargs="*", default=None, help="只算这些 instance_id")
    args = parser.parse_args()

    t0 = time.time()
    from configs.config import load_config
    rollouts = int(load_config().get("runtime.eval_rollouts", 10))
    fshgrl = [str(c) for c in checkpoints("fshgrl_run*")]
    if not fshgrl:
        raise SystemExit("[FAIL] 未找到 fshgrl_run*/checkpoint_best.pt，请先运行 run_02")
    drl = {name: [str(c) for c in checkpoints(f"{tag}_run*")] for tag, name in BASELINE_TAGS.items()}
    workers = args.cpsat_workers or max(1, min(8, (os.cpu_count() or 1) // max(args.jobs, 1)))
    opts = {"cpsat_time_limit": args.cpsat_time_limit, "milp_time_limit": args.milp_time_limit,
            "online_time_limit": args.online_time_limit, "cpsat_workers": workers,
            "rollouts": rollouts, "fshgrl": fshgrl, "drl": drl}

    metas = read_index("small")
    if args.instances:
        metas = [m for m in metas if m["instance_id"] in set(args.instances)]
    finished = {r["instance_id"] for r in _read_rows()}
    todo = [m for m in metas if m["instance_id"] not in finished]
    step(f"small 档精确求解：{len(todo)} 个待算，{len(metas) - len(todo)} 个已在 {OUT.name} 中"
         f"（FSHGRL {len(fshgrl)} 个 run，jobs={args.jobs}，CP-SAT 线程={workers}，"
         f"MILP 时限={args.milp_time_limit:.0f}s）")

    def record(row):
        append_rows(OUT, [row], COLUMNS)
        print(f"  {row['instance_id']}: eta_off={row['eta_off']} ({row['eta_off_source']}) "
              f"CP-SAT {row['cpsat_status']} {row['cpsat_time_s']}s | MILP {row['milp_status']} "
              f"{row['milp_time_s']}s | 证书={row['cert_cpsat_in_milp']} 回放={row['replay_match']} | "
              f"online={row['eta_online']} FSHGRL={row['eta_fshgrl']} PDR={row['eta_best_pdr']}", flush=True)

    if args.jobs <= 1:
        for meta in todo:
            record(_evaluate_instance(meta, opts))
    else:
        # 显式用 spawn：Linux 与 Windows 行为一致，也避免 fork 继承 torch / OR-Tools 的线程状态
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(_evaluate_instance, meta, opts) for meta in todo]
            for future in as_completed(futures):
                record(future.result())

    rows = _read_rows()
    if rows:                                      # 按规模、交期重排一次，便于人读；内容不变
        order = {m["instance_id"]: k for k, m in enumerate(read_index("small"))}
        rows.sort(key=lambda r: order.get(r["instance_id"], len(order)))
        OUT.unlink()
        append_rows(OUT, rows, COLUMNS)
    n = len(rows)
    count = lambda key: sum(1 for r in rows if r.get(key) == "1")          # noqa: E731
    print(f"\n[P-MILPCHK] CP-SAT 排程回放一致：{count('replay_match')}/{n}", flush=True)
    print(f"[证书] CP-SAT 最优排程满足字面 MILP 全部约束：{count('cert_cpsat_in_milp')}/{n}", flush=True)
    both = sum(1 for r in rows if r["eta_off_source"] == "cpsat=milp")
    conflicts = sum(1 for r in rows if r["eta_off_source"] == "CONFLICT")
    milp_opt = sum(1 for r in rows if r["milp_status"] == "OPTIMAL")
    print(f"[交叉核对] CP-SAT 证得最优 {sum(1 for r in rows if r['cpsat_status'] == 'OPTIMAL')}/{n}，"
          f"HiGHS MILP 证得最优 {milp_opt}/{n}，二者都证得最优且相等 {both}/{n}，"
          f"CONFLICT {conflicts}", flush=True)
    print(f"[在线] 回放一致：{count('replay_match_online')}/{n}，每次重解都证得最优："
          f"{count('online_all_optimal')}/{n}", flush=True)
    if any(r["eta_off_source"] == "CONFLICT" for r in rows):
        print("[WARN] 存在 CONFLICT：两个求解器的结论互相矛盾，说明两份公式化之一有错", flush=True)
    done(t0, OUT)


if __name__ == "__main__":
    main()
