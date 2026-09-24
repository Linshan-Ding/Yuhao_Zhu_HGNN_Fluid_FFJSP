"""训练三个学习基线（规则选择 PPO、三元组 DQN、分层 DQN），与主方法同一环境、同一交互预算。

用法：python scripts/run_04_train_baselines.py [--only rulesel tdqn hdqn] [--runs 5]
"""
from _bootstrap import BASELINE_TAGS
from _train_matrix import train_tags

train_tags(BASELINE_TAGS, "训练学习基线")
