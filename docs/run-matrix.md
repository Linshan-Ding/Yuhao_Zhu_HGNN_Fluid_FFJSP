# 运行矩阵（v2）

第一轮正式矩阵（73 作业）已于 2026-10 完成并发布在 `result/formal/`（标签 `paper2-v1-formal`）。下表为 v2 的矩阵：先预注册试点，通过后启动正式 v2（`result/formal2/`）。动态状态以各根目录的 budget_ledger.json 和各作业 status.json 为准。

## 预注册试点（result/pilot，入口 scripts/run_pilot.py）

| 作业 | 方法 | 种子 | 预算 |
|---|---|---:|---:|
| full_s1 | full | 1 | 500000 |
| full_s2 | full | 2 | 500000 |
| full_s3 | full | 3 | 500000 |
| no_graph_s1 | no_graph | 1 | 500000 |
| no_graph_s2 | no_graph | 2 | 500000 |
| no_graph_s3 | no_graph | 3 | 500000 |
| no_scenario_s1 | no_scenario | 1 | 500000 |
| no_scenario_s2 | no_scenario | 2 | 500000 |
| no_scenario_s3 | no_scenario | 3 | 500000 |
| no_demo_s1 | no_demo | 1 | 500000 |
| no_demo_s2 | no_demo | 2 | 500000 |
| no_demo_s3 | no_demo | 3 | 500000 |
| hgnn_s1 | hgnn | 1 | 500000 |
| hgnn_s2 | hgnn | 2 | 500000 |
| hgnn_s3 | hgnn | 3 | 500000 |
| ppo_s1 | ppo | 1 | 500000 |
| ppo_s2 | ppo | 2 | 500000 |
| ppo_s3 | ppo | 3 | 500000 |

共 18 作业，900 万交互；七条规则零训练。通过门槛与裁决见 preregistration-v2.md。

## 单算例收敛研究（result/instance，入口 scripts/run_instance_study.py）

12 个争用离线参照算例（CS24/CS36-R1.3-K1.5|2.0-r1..3）× 6 配置（full、no_graph、no_scenario、no_demo、hgnn、ppo）× 3 种子，每作业 100000 交互，作业名 `<方法>_<算例>_s<种子>`（如 `full_CS24-R1.3-K1.5-r1_s1`），共 216 作业、2160 万交互；七条规则零训练。无通过门槛，见 preregistration-v2.md 第 8 节。

## 正式 v2（result/formal2）

| 作业 | 方法 | 种子 | 预算 | 入口 |
|---|---|---:|---:|---|
| full_s1 | full | 1 | 1000000 | run_02_train_main.py |
| full_s2 | full | 2 | 1000000 | run_02_train_main.py |
| full_s3 | full | 3 | 1000000 | run_02_train_main.py |
| full_s4 | full | 4 | 1000000 | run_02_train_main.py |
| full_s5 | full | 5 | 1000000 | run_02_train_main.py |
| dqn_s1 | dqn | 1 | 1000000 | run_03_train_comparators.py |
| dqn_s2 | dqn | 2 | 1000000 | run_03_train_comparators.py |
| dqn_s3 | dqn | 3 | 1000000 | run_03_train_comparators.py |
| dqn_s4 | dqn | 4 | 1000000 | run_03_train_comparators.py |
| dqn_s5 | dqn | 5 | 1000000 | run_03_train_comparators.py |
| ddqn_s1 | ddqn | 1 | 1000000 | run_03_train_comparators.py |
| ddqn_s2 | ddqn | 2 | 1000000 | run_03_train_comparators.py |
| ddqn_s3 | ddqn | 3 | 1000000 | run_03_train_comparators.py |
| ddqn_s4 | ddqn | 4 | 1000000 | run_03_train_comparators.py |
| ddqn_s5 | ddqn | 5 | 1000000 | run_03_train_comparators.py |
| a2c_s1 | a2c | 1 | 1000000 | run_03_train_comparators.py |
| a2c_s2 | a2c | 2 | 1000000 | run_03_train_comparators.py |
| a2c_s3 | a2c | 3 | 1000000 | run_03_train_comparators.py |
| a2c_s4 | a2c | 4 | 1000000 | run_03_train_comparators.py |
| a2c_s5 | a2c | 5 | 1000000 | run_03_train_comparators.py |
| ppo_s1 | ppo | 1 | 1000000 | run_03_train_comparators.py |
| ppo_s2 | ppo | 2 | 1000000 | run_03_train_comparators.py |
| ppo_s3 | ppo | 3 | 1000000 | run_03_train_comparators.py |
| ppo_s4 | ppo | 4 | 1000000 | run_03_train_comparators.py |
| ppo_s5 | ppo | 5 | 1000000 | run_03_train_comparators.py |
| hgnn_s1 | hgnn | 1 | 1000000 | run_03_train_comparators.py |
| hgnn_s2 | hgnn | 2 | 1000000 | run_03_train_comparators.py |
| hgnn_s3 | hgnn | 3 | 1000000 | run_03_train_comparators.py |
| hgnn_s4 | hgnn | 4 | 1000000 | run_03_train_comparators.py |
| hgnn_s5 | hgnn | 5 | 1000000 | run_03_train_comparators.py |
| dual_attention_s1 | dual_attention | 1 | 1000000 | run_03_train_comparators.py |
| dual_attention_s2 | dual_attention | 2 | 1000000 | run_03_train_comparators.py |
| dual_attention_s3 | dual_attention | 3 | 1000000 | run_03_train_comparators.py |
| dual_attention_s4 | dual_attention | 4 | 1000000 | run_03_train_comparators.py |
| dual_attention_s5 | dual_attention | 5 | 1000000 | run_03_train_comparators.py |
| no_graph_s1 | no_graph | 1 | 1000000 | run_03_train_comparators.py |
| no_graph_s2 | no_graph | 2 | 1000000 | run_03_train_comparators.py |
| no_graph_s3 | no_graph | 3 | 1000000 | run_03_train_comparators.py |
| no_graph_s4 | no_graph | 4 | 1000000 | run_03_train_comparators.py |
| no_graph_s5 | no_graph | 5 | 1000000 | run_03_train_comparators.py |
| no_scenario_s1 | no_scenario | 1 | 1000000 | run_03_train_comparators.py |
| no_scenario_s2 | no_scenario | 2 | 1000000 | run_03_train_comparators.py |
| no_scenario_s3 | no_scenario | 3 | 1000000 | run_03_train_comparators.py |
| no_scenario_s4 | no_scenario | 4 | 1000000 | run_03_train_comparators.py |
| no_scenario_s5 | no_scenario | 5 | 1000000 | run_03_train_comparators.py |
| no_demo_s1 | no_demo | 1 | 1000000 | run_03_train_comparators.py |
| no_demo_s2 | no_demo | 2 | 1000000 | run_03_train_comparators.py |
| no_demo_s3 | no_demo | 3 | 1000000 | run_03_train_comparators.py |
| no_demo_s4 | no_demo | 4 | 1000000 | run_03_train_comparators.py |
| no_demo_s5 | no_demo | 5 | 1000000 | run_03_train_comparators.py |
| impact_only_s1 | impact_only | 1 | 1000000 | run_03_train_comparators.py |
| impact_only_s2 | impact_only | 2 | 1000000 | run_03_train_comparators.py |
| impact_only_s3 | impact_only | 3 | 1000000 | run_03_train_comparators.py |
| impact_only_s4 | impact_only | 4 | 1000000 | run_03_train_comparators.py |
| impact_only_s5 | impact_only | 5 | 1000000 | run_03_train_comparators.py |
| sensitivity_kmax4_s1 | full | 1 | 500000 | run_04_train_sensitivity.py |
| sensitivity_kmax4_s2 | full | 2 | 500000 | run_04_train_sensitivity.py |
| sensitivity_kmax4_s3 | full | 3 | 500000 | run_04_train_sensitivity.py |
| sensitivity_kmax16_s1 | full | 1 | 500000 | run_04_train_sensitivity.py |
| sensitivity_kmax16_s2 | full | 2 | 500000 | run_04_train_sensitivity.py |
| sensitivity_kmax16_s3 | full | 3 | 500000 | run_04_train_sensitivity.py |
| sensitivity_t03_s1 | full | 1 | 500000 | run_04_train_sensitivity.py |
| sensitivity_t03_s2 | full | 2 | 500000 | run_04_train_sensitivity.py |
| sensitivity_t03_s3 | full | 3 | 500000 | run_04_train_sensitivity.py |
| sensitivity_t07_s1 | full | 1 | 500000 | run_04_train_sensitivity.py |
| sensitivity_t07_s2 | full | 2 | 500000 | run_04_train_sensitivity.py |
| sensitivity_t07_s3 | full | 3 | 500000 | run_04_train_sensitivity.py |
| sensitivity_w003_s1 | full | 1 | 500000 | run_04_train_sensitivity.py |
| sensitivity_w003_s2 | full | 2 | 500000 | run_04_train_sensitivity.py |
| sensitivity_w003_s3 | full | 3 | 500000 | run_04_train_sensitivity.py |
| sensitivity_w03_s1 | full | 1 | 500000 | run_04_train_sensitivity.py |
| sensitivity_w03_s2 | full | 2 | 500000 | run_04_train_sensitivity.py |
| sensitivity_w03_s3 | full | 3 | 500000 | run_04_train_sensitivity.py |

共73作业，6400万交互。CPU单进程一线程，上限8并发；初始化与情景已包含于作业预算。敏感性单因素：kmax4/kmax16（最大未来数）、t03/t07（可靠性阈值）、w003/w03（辅助权重）。
