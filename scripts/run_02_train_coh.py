"""训练主方法 CoH：n_runs 个独立 run（种子 1..n），每个 run 训练 training.total_steps 环境步。

用法：python scripts/run_02_train_coh.py [--runs 5] [--total-steps N] [--workers W]
中断后重跑会从 checkpoint_last.pt 续跑；已训满预算的 run 跳过。
"""
from _train_matrix import train_tags

train_tags(["coh"], "训练主方法")
