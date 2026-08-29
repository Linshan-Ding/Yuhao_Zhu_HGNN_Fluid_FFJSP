"""训练全部消融变体。新增变体 = 在 VARIANTS 里加一行，README 命令不变。

变体定义见 configs/ablation/*.yaml 与论文 Table T-NEW-2 的设计矩阵。
FSHGRL-RP / FSHGRL-HP 是同基数对照，分离"宏观引导"与"单纯缩小动作空间"。
"""
import time

from _bootstrap import ROOT, done, run_py, step

N_RUNS = 5
EPOCHS = 1000
VARIANTS = {
    "noff":  ["ablation/noff.yaml"],
    "nofp":  ["ablation/nofp.yaml"],
    "rp":    ["ablation/rp.yaml"],
    "hp":    ["ablation/hp.yaml"],
    "nofa":  ["ablation/nofa.yaml"],
    "nosa":  ["ablation/nosa.yaml"],
    "nohg":  ["ablation/nohg.yaml"],
    "noall": ["ablation/noall.yaml"],
    # 以下三个变体不动五机制网格，各改一处设计决策
    "maxmin": ["ablation/fluid_maxmin.yaml"],   # 流体目标：max-min 交期可行比 vs 吞吐对齐
    "nonoop": ["ablation/nonoop.yaml"],         # 关闭主动空闲，退回 non-delay 策略类
    "nobc":   ["ablation/nobc.yaml"],           # 关闭规则热启动，从随机初始化训练
}

for tag, extra in VARIANTS.items():
    for i in range(1, N_RUNS + 1):
        name = f"{tag}_run{i}"
        ckpt = ROOT / "result" / name / "checkpoint_best.pt"
        if ckpt.exists():
            print(f"[SKIP] {name} 已完成")
            continue
        t0 = time.time()
        step(f"训练消融变体 {name}")
        run_py("train.py", "--config", *extra, "--run-name", name, "--epochs", EPOCHS)
        done(t0, ckpt)
