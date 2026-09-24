"""冒烟用：在 2 个最小算例上走通精确参照的全部路径，任何一项不一致即报错退出。

CP-SAT 离线最优 -> 离散事件环境回放 -> 滚动精确重优化 -> 在线排程回放。
"""
from _bootstrap import ROOT  # noqa: F401

from configs.config import load_config
from data.dataset import read_index
from data.generator import load_instance_csv
from environment.problem import Problem
from exact.cpsat import replay_check, solve_cpsat, solve_online_reoptimization

cfg = load_config(["coh.yaml"])
rows = sorted(read_index("small"), key=lambda r: int(r["S"]))[:2]
for row in rows:
    problem = Problem(load_instance_csv(row["path"], row["tier"], row["instance_id"]))
    result = solve_cpsat(problem, time_limit_s=60, workers=2)
    check = replay_check(problem, result, cfg)
    online = solve_online_reoptimization(problem, time_limit_per_solve=30)
    online_check = replay_check(problem, online, cfg)
    print(f"  {row['instance_id']}: eta_off={result.eta:.4f} ({result.status}, {result.seconds:.2f}s) "
          f"回放={check['match']} 在线={online.eta:.4f}（{online.n_solves} 次重解）在线回放={online_check['match']}",
          flush=True)
    if not check["match"]:
        raise SystemExit(f"[FAIL] 离线排程回放不一致：{check}")
    if not online_check["match"] or online.eta > result.eta + 1e-9:
        raise SystemExit(f"[FAIL] 在线重优化排程不一致或超过离线最优：{online_check}")
print("  精确参照与仿真器描述同一问题（两份排程回放一致，在线 <= 离线）", flush=True)
