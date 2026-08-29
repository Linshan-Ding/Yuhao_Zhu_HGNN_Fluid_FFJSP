"""训练三个学习基线（等交互预算、同一超参搜索协议）。

预算按**环境交互步数**对齐而非 epoch —— 各方法 episode 长度不同，按 epoch 对齐
不是等预算。规则基线无需训练，直接在 run_05 里评测。
"""
import time

from _bootstrap import ROOT, done, run_py, step, training_budget

N_RUNS = 5
EPOCHS = training_budget()   # 唯一真源：configs/algo.yaml ppo.total_epochs
BASELINES = ["DRLG", "AHP-DQN", "HSDDQN"]

for method in BASELINES:
    for i in range(1, N_RUNS + 1):
        tag = method.lower().replace("-", "")
        name = f"{tag}_run{i}"
        ckpt = ROOT / "result" / name / "checkpoint_best.pt"
        if ckpt.exists():
            print(f"[SKIP] {name} 已完成")
            continue
        t0 = time.time()
        step(f"训练基线 {method}（{i}/{N_RUNS}）")
        run_py("scripts/_train_baseline.py", "--method", method,
               "--run-name", name, "--epochs", EPOCHS)
        done(t0, ckpt)
