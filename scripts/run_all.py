"""顺序执行 01-13 全部实验。任一步失败即停；修好后重跑会自动跳过已完成步骤。"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "run_01_prepare_data.py",
    "run_02_train_main.py",
    "run_03_train_ablations.py",
    "run_04_train_baselines.py",
    "run_05_eval_main.py",
    "run_06_pruning_analysis.py",
    "run_07_exact_optimality.py",
    "run_08_arrival_ood.py",
    "run_09_reward_exploration.py",
    "run_10_case_study.py",
    "run_11_aggregate_stats.py",
    "run_12_make_figures.py",
    "run_13_fill_placeholders.py",
]

for name in STEPS:
    print("\n" + "#" * 72 + f"\n# {name}\n" + "#" * 72, flush=True)
    if subprocess.call([sys.executable, str(HERE / name)]) != 0:
        sys.exit(f"[FAIL] {name} 失败，后续步骤已中止")
print("\n[ALL OK] 全部实验数据已产出，见 result/ 与 result/paper_values.tex", flush=True)
