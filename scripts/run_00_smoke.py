"""分钟级全链路冒烟：算例 -> 奖励恒等式 -> 多进程训练链路 -> 精确参照 -> 统计 -> 论文占位符集合。

投入完整算力前先跑这一条。任何一环断掉都会在这里暴露，而不是在几小时训练之后。
端到端的迷你全流程（含评测、统计、图、回填）见 `python scripts/run_all.py --smoke`。
"""
import time

from _bootstrap import ROOT, done, run_py, step

t0 = time.time()

step("1/6 生成固定算例（固定种子，重建结果相同）")
run_py("data/dataset.py")

step("2/6 校验奖励恒等式 sum_t r_t == eta（多档算例 + 训练采样）")
run_py("analysis/identity_check.py")

step("3/6 多进程训练链路：批量前向一致性、worker 往返、PPO 与 DQN 更新、可复现性")
run_py("scripts/_smoke_train_mp.py")

step("4/6 精确参照 + 回放校验（2 个最小算例）")
run_py("scripts/_smoke_exact.py")

step("5/6 统计模块自检")
run_py("scripts/_smoke_stats.py")

step("6/6 论文占位符集合 == run_09 SOURCES（论文仓库在同级目录时）")
if (ROOT.parent / "Yuhao_Zhu_FFJSP_Order_GNN_Fluid_model" / "main.tex").exists():
    run_py("scripts/_check_placeholders.py")
else:
    print("  [跳过] 未找到同级论文仓库 Yuhao_Zhu_FFJSP_Order_GNN_Fluid_model", flush=True)

done(t0, ROOT / "data" / "instances" / "index.csv")
print("\n[SMOKE OK] 全链路通过，可以开始正式实验（见 README §4 起）", flush=True)
