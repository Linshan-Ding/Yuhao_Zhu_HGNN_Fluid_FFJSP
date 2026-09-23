"""奖励配置与行为策略修正的消融（论文 Table T-NEW-9 三个面板）。

预算与主方法相同（configs/algo.yaml 的 ppo.total_epochs）。表中与主配置相同的四行
（beta_f=0、beta_Psi=主配置值、kappa_d=1、修正比率）直接复用 fshgrl_run1-5，不重训；
其余 9 个配置各训练一个 run：
  面板 (a) beta_f in {0.05, 0.1, 0.2, 0.5}，beta_Psi 同主配置（beta_f>0 会改变最优策略）；
  面板 (b) beta_Psi=0、kappa_d=0、比率差奖励（稿件 Eq. pathological，病态对照）；
  面板 (c) 未修正比率、纯 on-policy。

一个 run 只有当 log.csv 记满预算的 epoch 数才算完成；中断的 run 会从头重训
（train.py 覆盖同名目录），不会被当成已完成而跳过。清单外的 rw_* 目录一律忽略。

用法：
  python scripts/run_09_reward_exploration.py                 # 训练缺失/未完成的 run，再汇总
  python scripts/run_09_reward_exploration.py --only rw_betaf0p05 rw_kappad0   # 只训练这几个（可多开并行）
  python scripts/run_09_reward_exploration.py --collect       # 只汇总
汇总：result/reward_exploration.csv（见 scripts/_collect_reward_ablation.py）。
"""
import argparse
import csv
import time

from _bootstrap import ROOT, done, run_py, step, training_budget

EPOCHS = training_budget()
REUSE = "fshgrl"          # 与主配置相同的行：复用 fshgrl_run1-5 的训练日志与 eval_results.csv

# (面板, 表中行, run 目录名或 REUSE, 训练参数)
ROWS = [
    ("(a) fluid-alignment weight", "beta_f=0", REUSE, None),
    ("(a) fluid-alignment weight", "beta_f=0.05", "rw_betaf0p05", ["--set", "reward.fluid_align_weight=0.05"]),
    ("(a) fluid-alignment weight", "beta_f=0.1", "rw_betaf0p1", ["--set", "reward.fluid_align_weight=0.1"]),
    ("(a) fluid-alignment weight", "beta_f=0.2", "rw_betaf0p2", ["--set", "reward.fluid_align_weight=0.2"]),
    ("(a) fluid-alignment weight", "beta_f=0.5", "rw_betaf0p5", ["--set", "reward.fluid_align_weight=0.5"]),
    ("(b) shaping and discard weight", "beta_psi=0", "rw_betapsi0", ["--set", "reward.potential_weight=0.0"]),
    ("(b) shaping and discard weight", "beta_psi=main", REUSE, None),
    ("(b) shaping and discard weight", "kappa_d=0", "rw_kappad0", ["--set", "reward.discard_weight=0.0"]),
    ("(b) shaping and discard weight", "kappa_d=1", REUSE, None),
    ("(b) shaping and discard weight", "ratio_difference", "rw_ratiodiff", ["--config", "ablation/ratiodiff.yaml"]),
    ("(c) importance ratio", "uncorrected", "rw_uncorrected", ["--config", "ablation/uncorrected.yaml"]),
    ("(c) importance ratio", "corrected", REUSE, None),
    ("(c) importance ratio", "onpolicy", "rw_onpolicy", ["--config", "ablation/onpolicy.yaml"]),
]
TRAINED = [name for _, _, name, _ in ROWS if name != REUSE]


def logged_epochs(name):
    log = ROOT / "result" / name / "log.csv"
    if not log.exists():
        return 0
    with log.open(encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def is_complete(name):
    return (logged_epochs(name) >= EPOCHS
            and (ROOT / "result" / name / "checkpoint_best.pt").exists())


def train(name, extra):
    t0 = time.time()
    have = logged_epochs(name)
    step(f"{name}：{' '.join(extra)}（预算 {EPOCHS} epoch"
         + (f"；已有 {have} epoch 的中断记录，从头重训" if have else "") + "）")
    if extra[0] == "--set":
        run_py("scripts/_train_override.py", "--run-name", name, "--epochs", EPOCHS, *extra)
    else:
        run_py("train.py", *extra, "--run-name", name, "--epochs", EPOCHS)
    done(t0, ROOT / "result" / name / "checkpoint_best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=TRAINED, default=None,
                        help="只训练这些 run，不汇总（便于多开终端并行）")
    parser.add_argument("--collect", action="store_true", help="只汇总，不训练")
    args = parser.parse_args()

    if not args.collect:
        for _, _, name, extra in ROWS:
            if name == REUSE or (args.only and name not in args.only):
                continue
            if is_complete(name):
                print(f"[SKIP] {name} 已训满 {EPOCHS} epoch", flush=True)
                continue
            train(name, extra)
        if args.only:
            return

    t0 = time.time()
    step("汇总三个面板 -> result/reward_exploration.csv")
    run_py("scripts/_collect_reward_ablation.py")
    done(t0, ROOT / "result" / "reward_exploration.csv")


if __name__ == "__main__":
    main()
