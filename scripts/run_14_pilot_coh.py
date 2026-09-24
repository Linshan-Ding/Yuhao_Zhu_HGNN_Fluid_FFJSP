"""CoH（Commit-or-Hold）试点训练：新论文 docs/experiment-spec.md §9。

四个配置：P0 最小骨干 / P1 + 承诺评估器 / P2 + 等待门控 / P3 第一波胜者 + 纯 on-policy。
每配置 3 个独立 run，与主方法相同的 epoch 预算。完成判据是 log.csv 跑满预算，不是"有
checkpoint 就算完成"。run 名 coh_p{k}_run{i} 被 _bootstrap.checkpoints() 排除，不会混进
run_05/run_11 的主方法分析。试点只用于选配置，最终矩阵另起 run。

用法：python scripts/run_14_pilot_coh.py [--only p0 p1 p2 p3] [--runs 3]
每个训练进程默认单线程（OMP_NUM_THREADS=1），几个配置并行开跑即可。
"""
import argparse
import csv
import os
import time

from _bootstrap import ROOT, done, run_py, step, training_budget

# 每个训练进程只用一个线程。网络很小（embed 16、MLP 128），多线程 BLAS 没有收益；
# 而几个 run 并行时各开满线程会互相自旋等待，实测能慢一个数量级。用户显式设了就尊重。
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

PILOT = {"p0": "coh/p0_backbone.yaml", "p1": "coh/p1_critic.yaml",
         "p2": "coh/p2_gate.yaml", "p3": "coh/p3_onpolicy.yaml"}
EPOCHS = training_budget()


def logged_epochs(name: str) -> int:
    path = ROOT / "result" / name / "log.csv"
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=list(PILOT), default=["p0", "p1", "p2"])
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    for tag in args.only:
        for i in range(1, args.runs + 1):
            name = f"coh_{tag}_run{i}"
            if logged_epochs(name) >= EPOCHS:
                print(f"[SKIP] {name} 已训满 {EPOCHS} epoch", flush=True)
                continue
            t0 = time.time()
            step(f"试点 {tag.upper()}（configs/{PILOT[tag]}）第 {i}/{args.runs} 次 run")
            run_py("train.py", "--config", PILOT[tag], "--run-name", name, "--epochs", EPOCHS)
            done(t0, ROOT / "result" / name / "checkpoint_best.pt")


if __name__ == "__main__":
    main()
