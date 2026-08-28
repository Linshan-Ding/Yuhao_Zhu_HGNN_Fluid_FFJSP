"""冒烟用：在 2 个最小算例上跑精确解并校验回放一致性。"""
from _bootstrap import ROOT  # noqa: F401

from data.dataset import read_index
from data.generator import load_instance_csv
from environment.problem import Problem
from exact.milp import replay_check, solve_cpsat

rows = sorted(read_index("small"), key=lambda r: int(r["S"]))[:2]
for row in rows:
    problem = Problem(load_instance_csv(row["path"], row["tier"], row["instance_id"]))
    result = solve_cpsat(problem, time_limit_s=60)
    check = replay_check(problem, result)
    print(f"  {row['instance_id']}: eta_off={result.eta:.4f} ({result.status}, "
          f"{result.seconds:.2f}s) replay_match={check['match']}", flush=True)
    if not check["match"]:
        raise SystemExit(f"[FAIL] 解回放不一致：{check}")
print("  精确解与仿真器描述同一问题（回放一致）", flush=True)
