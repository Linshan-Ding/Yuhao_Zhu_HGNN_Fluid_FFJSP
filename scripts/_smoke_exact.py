"""冒烟用：在 2 个最小算例上走通精确求解的全部路径，任何一项不一致即报错退出。

CP-SAT 离线最优 -> HiGHS MILP 交叉核对 -> MILP 证书 -> 离散事件环境回放 -> 在线重优化。
"""
from _bootstrap import ROOT  # noqa: F401

from configs.config import load_config
from data.dataset import read_index
from data.generator import load_instance_csv
from environment.problem import Problem
from exact.milp import (milp_certificate, replay_check, solve_cpsat, solve_milp,
                        solve_online_reoptimization)

cfg = load_config()
rows = sorted(read_index("small"), key=lambda r: int(r["S"]))[:2]
for row in rows:
    problem = Problem(load_instance_csv(row["path"], row["tier"], row["instance_id"]))
    result = solve_cpsat(problem, time_limit_s=60)
    check = replay_check(problem, result, cfg)
    cert = milp_certificate(problem, result)
    milp = solve_milp(problem, time_limit_s=120)
    online = solve_online_reoptimization(problem, time_limit_per_solve=30)
    online_check = replay_check(problem, online, cfg)
    print(f"  {row['instance_id']}: eta_off={result.eta:.4f} ({result.status}, {result.seconds:.2f}s) "
          f"MILP={milp.eta:.4f} ({milp.status}, {milp.seconds:.1f}s) 证书={cert['ok']} "
          f"回放={check['match']} 在线={online.eta:.4f} 在线回放={online_check['match']}", flush=True)
    if not check["match"]:
        raise SystemExit(f"[FAIL] 解回放不一致：{check}")
    if not cert["ok"]:
        raise SystemExit(f"[FAIL] CP-SAT 排程不满足字面 MILP：{cert}")
    if milp.status == "OPTIMAL" and result.status == "OPTIMAL" and milp.n_completed != result.n_completed:
        raise SystemExit(f"[FAIL] 两个求解器都证得最优却不相等：CP-SAT {result.eta} vs MILP {milp.eta}")
    if not online_check["match"] or online.eta > result.eta + 1e-9:
        raise SystemExit(f"[FAIL] 在线重优化排程不一致或超过离线最优：{online_check}")
print("  精确解与仿真器描述同一问题（两个求解器一致、证书通过、回放一致）", flush=True)
