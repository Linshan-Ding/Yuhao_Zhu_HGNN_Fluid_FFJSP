# Commit-or-Hold — 超负荷柔性流水车间里非延迟调度的代价

本仓库是论文 *Learning to Hold Capacity: The Price of Non-Delay Dispatching in Overloaded Flexible
Flow Shops with Order Insertion* 的实验工程。**本 README 不是项目简介，而是复现手册**：把 §3–§8 的命令
按顺序执行，即可产出论文所需的全部数据；最后一步 `run_09` 把结果写成 `result/paper_values.tex` 与
`result/tables/*.tex`，任何占位符缺数据都会**报错列出缺口**。

---

## 0. 进度与运行顺序

| 步骤 | 脚本 | 产物 | 耗时（16 核 + RTX 5070 Ti 的估计；按第一个 epoch 的 `sps_collect` 重估） |
|---|---|---|---|
| 冒烟 | `run_00_smoke.py` | — | 3–5 分钟 |
| 数据 | `run_01_prepare_data.py` | `data/instances/`（已随仓库提交，重建结果相同） | 秒级 |
| 主方法 | `run_02_train_coh.py` | `result/coh_run1..5/` | 每 run 约 20–40 分钟（6M 步，14 worker） |
| 消融 | `run_03_train_ablations.py` | `result/coh_<变体>_run1..5/`（6 个变体） | 30 个 run，约 12–20 小时 |
| 基线 | `run_04_train_baselines.py` | `result/{rulesel,tdqn,hdqn}_run1..5/` | 15 个 run，约 8–14 小时（两个 DQN 基线每个 epoch 做 `0.05 × 新转移数` 步梯度，更新耗时约为 PPO 的 10 倍，看 `log.csv` 的 `t_update`） |
| 评测 | `run_05_eval.py --jobs 12` | `result/eval_results.csv`、`result/gate_trace.csv` | 20–40 分钟 |
| 精确参照 | `run_06_exact_reference.py --jobs 6` | `result/exact_results.csv` | 30–90 分钟（可断点续跑） |
| 统计 | `run_07_stats.py` | `result/*.csv`（见 §10） | 1 分钟 |
| 绘图 | `run_08_figures.py` | `result/figures/F1–F6` | 1 分钟 |
| 回填 | `run_09_fill_placeholders.py --paper-dir <论文目录>` | `paper_values.tex`、`tables/*.tex`、`figures/*.pdf`（复制到论文目录） | 秒级 |

一条命令跑完全部：`python scripts/run_all.py --jobs 12`（训练脚本会跳过已训满预算的 run，评测与统计每次重算）。
先用 `python scripts/run_all.py --smoke` 走一遍极小预算的全流程（约半小时），确认链路无误再投入算力。

## 1. 问题假设

- 柔性流水车间，R 种产品各走 J 个阶段，每阶段多台可选机器，工时为整数；订单按 Poisson 过程持续插入
  （评测另含 MMPP 与确定性到达）。
- 决策时点 = 至少一台机器空闲且至少一道工序就绪。动作 = (工序类型, 机器, 订单) 或"保留产能"（no-op）。
  no-op 只在存在更晚事件且连续次数 < 3 时可用（防死锁）。
- 每工序类型只暴露 top-5 张订单，可救的（临界比 ≥ 1.5）优先，再按交期。规则与学习策略看到同一候选集。
- 剩余路径最短工时超过交期的订单被丢弃；超期完工按未达成计。指标 η = 按时完工数 / 订单数。
- 奖励 r_t = ΔN_c / S，Σr = η（`run_00` 校验）。观测只用当前时刻已知的量（详见 `docs/experiment-spec.md` §1）。

## 2. 环境配置

```
conda create -n coh python=3.11 -y && conda activate coh
pip install -r requirements.txt
```

- GPU 可选：主进程的 PPO/DQN 更新在 CUDA 上跑（`runtime.update_device: auto`），采样 worker 始终在 CPU。
  按 [pytorch.org](https://pytorch.org) 安装带 CUDA 的 torch；CPU-only 也能跑，只是更新更慢。
- worker 数默认 `CPU 核数 − 2`（16 核机器 14 个），可用 `--workers` 覆盖。每个 worker 单线程。
- 求解器全部免授权：精确参照用 OR-Tools CP-SAT。
- Windows：多进程用 spawn 启动方式，所有入口脚本都在 `if __name__ == "__main__":` 下，直接在 PowerShell
  或 PyCharm 里运行即可（§9）。

## 3. 冒烟自检（先跑这条，分钟级）

```
python scripts/run_00_smoke.py
```

六步：生成算例 → 奖励恒等式（24 个算例）→ 多进程训练链路（批量前向与单样本前向一致、worker 往返、PPO 与
DQN 更新、同种子可复现、贪心确定性、checkpoint 重载）→ 精确参照两个小算例（CP-SAT + 滚动重优化 + 回放一致）
→ 统计模块自检 → 论文 `\PH{}` 键集合与 `run_09` 的 SOURCES 逐一相等（同级目录有论文仓库时）。任何一项不过
都会报错退出。

## 4. 数据准备

```
python scripts/run_01_prepare_data.py
```

四档固定算例（grid 36 / val 6 / small 24 / ood 7），每档固定种子，重建逐位相同；`data/instances/` 已随仓库
提交，复现基准是这些文件本身。设计见 `configs/instance.yaml` 与 `docs/experiment-spec.md` §3。

## 5. 训练主方法

```
python scripts/run_02_train_coh.py            # 5 个 run，种子 1..5，每 run 6M 环境步
python scripts/run_02_train_coh.py --runs 3 --total-steps 3000000 --workers 12   # 缩小预算的写法
```

- 预算按环境交互步数计（`configs/algo.yaml` 的 `training.total_steps`），对所有方法相同。
- 每个 epoch 固定 56 条 episode（与 worker 数无关），每 2 个 epoch 在 val 档贪心验证一次，
  `checkpoint_best.pt` 按验证 η 刷新；`checkpoint_last.pt` 含优化器，中断后重跑同一命令即续跑。
- 训练曲线在 `result/coh_run1/log.csv`（列见 `docs/experiment-spec.md` §4）。看 `sps_collect`（采样步/秒）
  估算总耗时：`total_steps / sps_collect` 秒。

单独启动一个 run：`python train.py --config coh.yaml --run-name coh_run1 --seed 1`。

## 6. 基线与消融

```
python scripts/run_03_train_ablations.py      # 6 个消融变体 × 5 run
python scripts/run_04_train_baselines.py      # 3 个学习基线 × 5 run
```

| 变体 / 基线 | 配置 | 含义 |
|---|---|---|
| CoH-NoHold | `ablation/nohold.yaml` | 不给保留产能动作 = 同一网络的 non-delay 版本（"非延迟的代价"） |
| CoH-NoCritic | `ablation/nocritic.yaml` | 去掉承诺评估器 |
| CoH-NoGate | `ablation/nogate.yaml` | no-op 作为普通一行候选进 softmax |
| CoH-NoHoldCritic | `ablation/noholdcritic.yaml` | 门控不用等待前景评估器 |
| CoH-NoGateFeat | `ablation/nogatefeat.yaml` | 门控不看工况特征 |
| CoH-EDD | `ablation/edd_exposure.yaml` | 候选只按交期暴露 |
| RuleSel-PPO | `baseline/rulesel_ppo.yaml` | 规则选择 PPO（6 条规则 + hold） |
| Triplet-DQN | `baseline/triplet_dqn.yaml` | 三元组 Double DQN + 优先级代理特征 |
| Hier-DQN | `baseline/hier_dqn.yaml` | 分层 Double DQN（上层规则/hold，下层前 3 个候选） |

`--only` 可指定子集，例如 `python scripts/run_03_train_ablations.py --only coh_nohold coh_nogate`。
规则基线不训练，评测时直接计算；SPT-Idle 另扫阈值 {1.0, 1.25, 1.5, 2.0, 3.0}。

## 7. 评测与精确参照

```
python scripts/run_05_eval.py --jobs 12
python scripts/run_06_exact_reference.py --jobs 6
```

- `run_05`：grid 与 ood 两档，规则 + 全部已训练的 run，逐算例一行写 `result/eval_results.csv`；CoH 主方法
  另落盘每个决策的门控量 `result/gate_trace.csv`。重跑先清空再全部重算。
- `run_06`：small 档 24 个算例，CP-SAT 离线最优（限时 3600 s）、滚动精确重优化（每次重解限时 60 s）、
  两份排程回放进仿真器核对，再算 CoH、非延迟策略、最强规则、最强基线。逐算例落盘，可断点续跑。

## 8. 统计聚合、绘图与回填

```
python scripts/run_07_stats.py
python scripts/run_08_figures.py
python scripts/run_09_fill_placeholders.py --paper-dir ../Yuhao_Zhu_FFJSP_Order_GNN_Fluid_model
```

- `run_07` 按 `docs/experiment-spec.md` §7 的预注册判据机械核对，打印裁决表并写 `result/verdict.csv`；
  另产出池化与分层比较、每格均值、异质性、方差分解、Friedman、预算核对、校准、OOD 与精确参照汇总。
- `run_09` 把数值写成 `paper_values.tex`、把整张数据表写成 `tables/*.tex`，`--paper-dir` 直接复制到论文
  目录；有占位符缺数据即非零退出并列出缺口。论文里 `\PH{}` 的键集合与脚本 `SOURCES` 逐一相等，核对命令见
  论文仓库 README。

## 9. 在 PyCharm / Windows 中手动启动

- 把仓库根目录设为工作目录（Run configuration 的 Working directory），逐个运行 `scripts/run_XX_*.py`。
- 并行：开多个终端标签页分别运行 `run_03 --only ...` 的不同子集，或在一个训练里加大 `--workers`。
  一次只跑一个训练时用满 `CPU 核数 − 2` 个 worker；两个训练并行时各分一半。
- 中断的训练重跑同一命令即从 `checkpoint_last.pt` 续跑。

## 10. 产物对照表（终检）

| 文件 | 论文去向 |
|---|---|
| `data/instances/index.csv` | 算例表 `tables/tab_instances.tex`、N-GRID/N-VAL/N-SMALL/N-OOD |
| `result/eval_results.csv` | 主结果、分层、每格均值、保留产能（经 run_07） |
| `result/gate_trace.csv` | 门控决策图 F3、校准图 F4、BRIER/ECE |
| `result/exact_results.csv` | 精确参照表 `tab_exact.tex`、ETA-OFF/ETA-ONLINE/GAP-* |
| `result/stats_summary.csv` | 规则表、基线表、消融表的差值/胜场/Holm p/δ 列，GAIN-/W-/P-/D-* |
| `result/stratified_summary.csv` | 分层表（论文用 PRICE-*/GAINR-*/W-*/P-* 三十个占位符拼成 `tab:price_bands`）、P-IUT；`tab_stratified.tex` 是含 SPT-Idle* 行的补充版本 |
| `result/cell_means.csv` | `tab_price.tex`、F1、F2、PRICE-MAX/MIN |
| `result/heterogeneity.csv` | P-HET-DDT、P-HET-RHO、HET-DDT |
| `result/budget_check.csv` | `tab_budget.tex`、R-RUNS、STEPS-RUN-M、T-RUN-MIN、SPS、BEST-STEPS-M、VAL-RANGE |
| `result/variance_decomposition.csv`、`friedman_nemenyi.csv` | ICC-SEED/ICC-INST、FRIEDMAN-P/CD、F6 |
| `result/ood_summary.csv` | `tab_ood.tex`、ETA-*-OOD、W-COH-OOD |
| `result/verdict.csv` | 摘要、§5 结果与结论的措辞依据（先读裁决，再填数） |
| `result/figures/F1–F6.pdf` | 论文图 |
| `result/paper_values.tex`、`result/tables/*.tex` | 复制到论文目录后 `\InputIfFileExists` 读入 |

## 目录结构

```
agent/        batch.py（填充批）networks.py（编码器与 CoH 网络）buffer.py（episode 记录、GAE、replay）
              workers.py（采样进程）ppo.py（批量 PPO）dqn.py（Double DQN）baselines/（规则、规则选择 PPO、DQN 网络）
analysis/     stats.py（检验与效应量）identity_check.py（奖励恒等式）
configs/      instance.yaml env.yaml algo.yaml coh.yaml ablation/ baseline/
data/         generator.py dataset.py instances/
environment/  env.py problem.py
exact/        cpsat.py（CP-SAT 离线最优、滚动重优化、回放校验）
result/       logger.py + 全部实验产物
scripts/      run_00–run_09、run_all、_bootstrap、_train_matrix、_smoke_*
paper_assets/ scripts/check.py（论文文体与一致性检查）
train.py eval.py
```
