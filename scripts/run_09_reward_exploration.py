"""奖励配置与行为策略修正的消融（论文 Table T-NEW-9 三个面板）。

面板 (a) 扫 beta_f（非势函数对齐项，beta_f>0 会改变最优策略——主实验恒取 0）；
面板 (b) 扫势函数塑形与丢弃权重，并含病态的比率差奖励作参照；
面板 (c) 对比未修正比率 / 修正比率 / 纯 on-policy，并记录实测最大比率与其理论上界。
"""
import time

from _bootstrap import ROOT, done, run_py, step

EPOCHS = 300          # 消融只需比较趋势，训练量小于主方法
PANEL_A = [0.0, 0.05, 0.1, 0.2, 0.5]
PANEL_B = [("beta_psi_0", "reward.potential_weight", 0.0),
           ("beta_psi_used", "reward.potential_weight", 0.1),
           ("kappa_d_0", "reward.discard_weight", 0.0),
           ("kappa_d_1", "reward.discard_weight", 1.0)]
PANEL_C = {"uncorrected": "ablation/uncorrected.yaml", "onpolicy": "ablation/onpolicy.yaml"}

for beta in PANEL_A:
    name = f"rw_betaf{str(beta).replace('.', 'p')}"
    if (ROOT / "result" / name / "checkpoint_best.pt").exists():
        print(f"[SKIP] {name}")
        continue
    t0 = time.time()
    step(f"面板(a) beta_f={beta}")
    run_py("scripts/_train_override.py", "--run-name", name, "--epochs", EPOCHS,
           "--set", f"reward.fluid_align_weight={beta}")
    done(t0)

for tag, key, value in PANEL_B:
    name = f"rw_{tag}"
    if (ROOT / "result" / name / "checkpoint_best.pt").exists():
        print(f"[SKIP] {name}")
        continue
    t0 = time.time()
    step(f"面板(b) {key}={value}")
    run_py("scripts/_train_override.py", "--run-name", name, "--epochs", EPOCHS,
           "--set", f"{key}={value}")
    done(t0)

for tag, config in PANEL_C.items():
    name = f"rw_{tag}"
    if (ROOT / "result" / name / "checkpoint_best.pt").exists():
        print(f"[SKIP] {name}")
        continue
    t0 = time.time()
    step(f"面板(c) {tag}")
    run_py("train.py", "--config", config, "--run-name", name, "--epochs", EPOCHS)
    done(t0)

step("汇总三个面板 -> result/reward_exploration.csv")
run_py("scripts/_collect_reward_ablation.py")
done(time.time(), ROOT / "result" / "reward_exploration.csv")
