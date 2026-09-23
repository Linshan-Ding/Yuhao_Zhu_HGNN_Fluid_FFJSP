"""顺序执行 01-13 全部实验。任一步失败即停。

重跑时的行为因脚本而异，不是"自动跳过已完成步骤"：
  * run_01 跳过已生成的算例档，run_02-04 跳过已有 checkpoint_best.pt 的训练 run；
  * run_07 只补 exact_results.csv 中缺失的算例，run_09 只训练未训满预算的 run；
  * run_05、06、08、10 会先清空自身产物再全部重算（数小时，且含随机 rollout 的数会变）；
  * run_11-13 只读结果文件，重算很快。
已有部分结果时，按 README §0 单独运行剩下的脚本，不要用本脚本从头重跑。
"""
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
