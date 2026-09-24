"""分钟级全链路冒烟：算例 -> 奖励恒等式 -> 训练 -> CoH 试点检查 -> 评测 -> 精确解 -> 统计。

投入完整算力前先跑这一条。任何一环断掉都会在这里暴露，而不是在几小时训练之后。
"""
import time

from _bootstrap import ROOT, done, run_py, step

t0 = time.time()

step("1/8 生成固定算例（已存在则跳过）")
run_py("data/dataset.py")

step("2/8 校验奖励恒等式 sum_t r_t == eta - kappa_d * nu")
run_py("analysis/identity_check.py")

step("3/8 微型训练（3 个 epoch）")
run_py("train.py", "--run-name", "smoke_run1", "--epochs", "3")

step("4/8 CoH 微型训练（p2 配置：承诺评估器 + 等待门控，2 个 epoch）")
run_py("train.py", "--config", "coh/p2_gate.yaml", "--run-name", "smoke_coh_run1", "--epochs", "2")

step("5/8 CoH 一致性与标签检查（旧 checkpoint 的评测不变；标签、归一化、held_share）")
run_py("scripts/_smoke_coh.py")

step("6/8 评测：一条规则 + 冒烟 checkpoint")
smoke_eval = ROOT / "result" / "smoke_eval.csv"
if smoke_eval.exists():
    smoke_eval.unlink()
run_py("eval.py", "--method", "EDD", "--variant", "EDD", "--run-id", "smoke",
       "--tiers", "main", "--rollouts", "1", "--out", "result/smoke_eval.csv")
run_py("eval.py", "--method", "FSHGRL", "--variant", "FSHGRL", "--run-id", "smoke",
       "--ckpt", "result/smoke_run1/checkpoint_best.pt",
       "--tiers", "main", "--rollouts", "1", "--out", "result/smoke_eval.csv")

step("7/8 精确解 + 解回放校验（2 个最小算例）")
run_py("scripts/_smoke_exact.py")

step("8/8 统计模块自检")
run_py("scripts/_smoke_stats.py")

done(t0, ROOT / "data" / "instances" / "index.csv",
     ROOT / "result" / "smoke_run1" / "log.csv", smoke_eval)
print("\n[SMOKE OK] 全链路通过，可以开始正式实验（见 README §4 起）", flush=True)
