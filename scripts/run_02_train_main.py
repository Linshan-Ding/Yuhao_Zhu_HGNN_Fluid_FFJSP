"""训练主方法 FSHGRL：N_RUNS 次独立 run（不设随机种子，每次训练即天然独立重复）。"""
import time

from _bootstrap import ROOT, done, run_py, step

N_RUNS = 5
EPOCHS = 1000            # 与 configs/algo.yaml 一致；改训练量请改 configs

for i in range(1, N_RUNS + 1):
    name = f"fshgrl_run{i}"
    ckpt = ROOT / "result" / name / "checkpoint_best.pt"
    if ckpt.exists():
        print(f"[SKIP] {name} 已完成")
        continue
    t0 = time.time()
    step(f"训练 {name}（第 {i}/{N_RUNS} 次独立 run）")
    run_py("train.py", "--run-name", name, "--epochs", EPOCHS)
    done(t0, ckpt, ROOT / "result" / name / "log.csv")
