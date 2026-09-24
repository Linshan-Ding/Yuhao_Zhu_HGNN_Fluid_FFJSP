"""按顺序跑完全部实验：数据 -> 训练（主方法、消融、基线）-> 评测 -> 精确参照 -> 统计 -> 图 -> 回填。

    python scripts/run_all.py [--runs 5] [--total-steps N] [--workers W] [--jobs J] [--smoke] [--from run_05]

--smoke 用极小预算走一遍全流程（约半小时，用来验证链路），产物不能用于论文。
训练脚本会跳过已训满预算的 run，评测与统计每次重算。
"""
import argparse

from _bootstrap import run_py, step

ORDER = ["run_01_prepare_data.py", "run_02_train_coh.py", "run_03_train_ablations.py",
         "run_04_train_baselines.py", "run_05_eval.py", "run_06_exact_reference.py",
         "run_07_stats.py", "run_08_figures.py", "run_09_fill_placeholders.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--from", dest="start", default=None, help="从某一步开始（脚本名前缀，如 run_05）")
    parser.add_argument("--paper-dir", default=None)
    args = parser.parse_args()
    runs = 1 if args.smoke else args.runs
    steps = 3000 if args.smoke else args.total_steps
    started = args.start is None
    for name in ORDER:
        if not started:
            started = name.startswith(args.start)
            if not started:
                continue
        step(name)
        extra = []
        if name.startswith(("run_02", "run_03", "run_04")):
            if runs:
                extra += ["--runs", runs]
            if steps:
                extra += ["--total-steps", steps]
            if args.workers:
                extra += ["--workers", args.workers]
        if name.startswith("run_04") and args.smoke:
            # 两个 DQN 基线每个 epoch 做 replay_ratio x 新转移数 步梯度；冒烟在 CPU 上只走通链路
            extra += ["--override", "dqn.replay_ratio=0.002"]
        if name.startswith(("run_05", "run_06")):
            extra += ["--jobs", args.jobs]
        if name.startswith("run_06") and args.smoke:
            extra += ["--cpsat-time-limit", 60, "--online-time-limit", 10]
        if name.startswith("run_09") and args.paper_dir:
            extra += ["--paper-dir", args.paper_dir]
        run_py(f"scripts/{name}", *extra)
    print("\n[ALL DONE] 全部步骤完成；论文数值在 result/paper_values.tex，表格在 result/tables/", flush=True)


if __name__ == "__main__":
    main()
