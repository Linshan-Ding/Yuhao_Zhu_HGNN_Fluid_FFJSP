"""训练全部消融变体（configs/ablation/*.yaml，RUN_SPECS 里以 coh_ 开头的项）。

用法：python scripts/run_03_train_ablations.py [--only coh_nohold coh_nocritic ...] [--runs 5]
"""
from _bootstrap import ABLATION_TAGS
from _train_matrix import train_tags

train_tags(ABLATION_TAGS, "训练消融变体")
