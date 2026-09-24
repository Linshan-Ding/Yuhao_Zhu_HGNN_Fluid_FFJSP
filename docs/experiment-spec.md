# 实验设计规格（Experiment Spec）— Commit-or-Hold / 超负荷柔性流水车间的非延迟代价

| 版本 | 日期 | 变更 |
|---|---|---|
| v2.0 | 2026-09-24 | 第二篇论文。第一篇（FSHGRL）已录用，其规格（v1.x）随代码一并移出工作树，只留 git 历史。 |

> 本文件是代码仓库的任务书。**论文里每一个 `\PH{}` 与每一张数据表都必须能在 §5 映射表里找到产出它的
> 脚本与 CSV 列**；反之每一列落盘都必须有论文去向。映射表无天窗 ⇔ 工程无天窗。

---

## 0. 范式判定

- **构造式 DRL**：PPO + 候选打分 actor-critic，动作是 (工序类型, 机器, 订单) 三元组或"保留产能"（no-op）。
- **环境不可张量化**（离散事件仿真，随机到达、完工事件、订单丢弃），因此采用**多进程采样 + 批量更新**：
  每个 worker 进程各一份环境与 CPU 策略，整条 episode 采完发回；主进程把一个 epoch 的转移填充成
  带掩码的批，在 GPU（有则用）上做 PPO 更新。spawn 启动方式，Windows 与 Linux 同一条代码路径。
- **预算按环境交互步数计**（`training.total_steps`），所有学习方法相同；epoch 的 episode 数固定，与
  worker 数无关，因此不同机器采到同一批数据。

## 1. 问题定义

- 动态柔性流水车间，订单持续插入（Poisson 到达，评测另含 MMPP 与确定性到达）。R 种产品各走 J 个阶段，
  每阶段多台可选机器，工时为整数（CP-SAT 用整数时间网格）。
- 决策时点：至少一台机器空闲且至少一道工序就绪。动作：派工三元组，或 no-op（保留产能：本时刻不派工，
  等到下一事件）。no-op 的两条防死锁守卫：(i) 存在严格更晚的未来事件；(ii) 连续 no-op 不超过 3 次。
- 候选暴露：每工序类型只暴露 top-K（K=5）张订单，顺序为"可救优先"（临界比 ≥ 1.5 的先暴露，再按交期）。
  规则与学习策略看到同一候选集。
- 订单在剩余路径最短工时已超过交期时被丢弃；超期完工按未达成计。目标 = 按时达成率 η = N_c / S。
- 奖励 r_t = ΔN_c / S，无任何塑形项与可调权重；恒等式 **Σ_t r_t = η** 由 `analysis/identity_check.py`
  在 24 个算例上校验（`run_00` 第 2 步）。
- 观测严格因果：只用当前时刻已知的量。时间除以 DDT 参数（交期政策），工时除以最大工时，计数除以已到达
  订单数；未到达订单的数量、交期、到达时间不进观测。

## 2. 方法：Commit-or-Hold（CoH）

- **骨干**：工序类型节点 [N,10] 与机器节点 [M,3] 各过逐节点 MLP（LayerNorm），掩码均值池化；候选特征 =
  [所选节点嵌入 ‖ 全局嵌入 ‖ 动作特征 [4] ‖ η_t]；actor 逐候选打分；critic 用全局嵌入 + 全局量 [5]。
- **承诺评估器** p̂(o|s)：逐候选"现在派出后按时完成"的概率，用同一 episode 内的实际结果做 BCE 监督；
  detach 后作为 actor 的一维特征。
- **等待门控**：P(hold) = σ(β·(logit ĥ − logit max p̂) + c(s))，ĥ 为等待前景评估器（被保留的机器下一次
  派工是否给了等待时尚未暴露且按时完成的订单，事后标签，BCE），c(s) 看 8 个工况特征 + 5 个全局量；
  P(a) = (1 − P(hold))·softmax(派工 logits)。门控里的 max p̂ 与 ĥ 不回传策略梯度（`coh.detach_gate_inputs`）。
  输出层小初始化、门控偏置 −1.5：起点是 non-delay（P(hold)≈0.18），再由训练学会何时等待。
- **训练**：PPO（clip 0.2，4 个 update-epoch，minibatch 2048/4096，熵 0.01，价值 0.5，梯度裁剪 0.5，
  lr 3e-4 线性退火到 0.1×），KL 用 k3 估计并早停（0.02，单批 4× 硬停），GAE λ 0.95，γ = 1；辅助损失
  权重 0.5。每个 epoch 56 条 episode，验证每 2 个 epoch，`checkpoint_best` 按验证 η。预算 6M 步，5 个
  独立 run（种子 1..5）；episode (epoch, k) 的算例与动作由 (种子, epoch, k) 决定，可精确复现。
- **贪心读出**：有 no-op 时先判 gate logit > 0 则等待，否则在派工行取 argmax。

## 3. 算例设计（`configs/instance.yaml`）

训练分布 `param_table`：R = J = 5、每阶段 5 台、工时 [25, 450]、S ∈ [20, 200]、ρ ∈ [0.8, 2.2]、
DDT ∈ [400, 2000]、逐单交期扰动 U[0.7, 1.4]、Poisson。评测档（每档固定种子，逐算例种子 = 档种子×1000 + 序号）：

| 档 | 设计 | 数 | 用途 |
|---|---|---|---|
| `grid` | ρ ∈ {0.8, 1.2, 1.6, 2.0} × DDT ∈ {700, 1100, 1800} × S ∈ {50, 100, 150} | 36 | 主评测、分层、代价热图 |
| `val` | ρ ∈ {1.0, 1.6, 2.0} × DDT ∈ {900, 1500}，S = 80 | 6 | 选 checkpoint，不进论文 |
| `small` | S ∈ {12, 16, 20, 25} × DDT ∈ {500, 600, 700} × 2，ρ = 2.5 | 24 | CP-SAT 离线最优与滚动重优化 |
| `ood` | S300、S500、每阶段 3 台、每阶段 8 台、R8J7、MMPP、确定性到达（ρ = 1.6、DDT = 900） | 7 | 不重训的迁移 |

## 4. 数据落盘清单

| 文件 | 产出脚本 | 列 |
|---|---|---|
| `data/instances/index.csv` | run_01 | instance_id, tier, path, S, R, J, M, machines_per_stage, DDT, rho_target, arrival_process, E_dt, Lambda, W_bar, p_bar, rho_sys, iota, regime, seed |
| `result/<run>/log.csv` | run_02–04（train.py） | iter, steps, episodes, eta_val, eta_train, return, ep_len, a_mean, noop_rate, p_hold, held_share, policy_loss, value_loss, entropy, approx_kl, clip_frac, ratio_max, update_epochs_done, commit_bce, commit_brier, hold_bce, n_labels, q_loss, q_mean, n_updates, epsilon, lr, collapse, sps_collect, sps_total, t_collect, t_update, t_val, elapsed_s |
| `result/eval_results.csv` | run_05 | instance_id, tier, S, DDT, rho_target, rho_sys, method, variant, run_id, seed, eta, nu, decision_time_ms, steps, n_cand_mean, noop_rate, noop_offer_rate, held_share |
| `result/gate_trace.csv` | run_05（CoH 主方法） | instance_id, variant, run_id, t, now, held, p_hold, p_max, hold_logit, gate_logit, gap, with_work, n_dispatch, exp_arrivals, backlog, viable, max_cr, min_proc, eta_t, open_share, rate, idle_share, p_hat_chosen, order, outcome |
| `result/exact_results.csv` | run_06 | instance_id, S, DDT, eta_off, cpsat_status, cpsat_time_s, replay_match, eta_online, online_solves, online_all_optimal, online_time_s, replay_match_online, eta_coh, eta_coh_sd, n_coh_runs, eta_nohold, eta_best_rule, best_rule, eta_best_drl, best_drl, gap_coh_off, gap_online_off |
| `result/stats_summary.csv` | run_07 | comparison, band, n, wins, ties, mean_diff, ci_lo, ci_hi, p_raw, p_holm, cliff_delta, eta_ours, eta_other |
| `result/stratified_summary.csv` | run_07 | 同上（band = DDT700/DDT1100/DDT1800/rho<1/rho>=1/pooled，含交–并检验行） |
| `result/cell_means.csv` | run_07 | rho, DDT, variant, n, eta, held_share, price_vs_nohold, price_vs_oracle_rule |
| `result/heterogeneity.csv` | run_07 | comparison, contrast, diff_of_means, p_perm |
| `result/variance_decomposition.csv`、`friedman_nemenyi.csv`、`budget_check.csv`、`ood_summary.csv`、`exact_summary.csv`、`calibration.csv`、`gate_map.csv`、`verdict.csv` | run_07 | 见脚本头 |
| `result/figures/F1–F6` | run_08 | 代价热图、保留产能、门控决策图、校准、学习曲线、临界差异图 |
| `result/paper_values.tex`、`result/tables/*.tex` | run_09 | 论文占位符与整张数据表 |

派生对手（run_07 计算）：`SPT-Idle*` = 逐算例在阈值 {1.0, 1.25, 1.5, 2.0, 3.0} 中取最优（事后选择，规则族的
上界）；`OracleRule` = 逐算例全部规则取最优；`BestRule` = 池化均值最高的单条规则。

## 5. claim → 实验 → 数据 → 论文占位符

| 论断 | 实验 | 数据 | 占位符 / 表 |
|---|---|---|---|
| 非延迟的代价：同一网络去掉等待后的损失 | run_05 grid（CoH vs CoH-NoHold） | stats_summary, stratified_summary, cell_means | GAIN-NOHOLD, W-NOHOLD, P-NOHOLD, D-NOHOLD, PRICE-*, PRICE-MAX/MIN, CELL-MAX/MIN, tab_price, F1 |
| 优于最强规则（含逐算例调优的 SPT-Idle*） | run_05 grid | stats_summary, stratified_summary | GAIN-SPTIDLESTAR, W-/P-/D-SPTIDLESTAR, GAIN-ORACLERULE …, P-IUT, GAINR-*, tab_rules |
| 优于三个学习基线 | run_04 + run_05 | stats_summary | ETA-RULESEL/TDQN/HDQN, NAME-BESTDRL, GAIN-/W-/P-/D-BESTDRL, tab_drl |
| 代价随交期紧度与负荷变化 | run_07 分层与异质性 | stratified_summary, heterogeneity | PRICE-/GAINR-/W-NOHOLD-/W-ORACLE-/P-NOHOLD-/P-ORACLE-{DDT700,DDT1100,DDT1800,RHOLT1,RHOGE1}（论文表 tab:price_bands）, P-HET-DDT, P-HET-RHO, HET-DDT；tab_stratified 为补充 |
| 策略保留了多少产能 | run_05 | eval_results.held_share | HELD-*, NOOP-RATE, F2 |
| 承诺评估器可校准 | run_05 trace | calibration | BRIER, ECE, F4 |
| 门控在何时等待 | run_05 trace | gate_map | F3 |
| 两个创新各自的价值 | run_03 + run_05 | stats_summary | A-NOCRITIC/P-NOCRITIC … A-EDD/P-EDD, tab_ablation |
| 与"优化完美但不能预判"的参照比 | run_06 small | exact_results, exact_summary | ETA-OFF, ETA-ONLINE, ETA-COH-SMALL, GAP-*, W-COH-ONLINE, P-COH-ONLINE, N-OPT, N-REPLAY, T-ONLINE, T-CPSAT, tab_exact |
| 不重训迁移到未见工况 | run_05 ood | ood_summary | ETA-COH-OOD, ETA-BESTRULE-OOD, W-COH-OOD, ETA-*-S500, ETA-*-MMPP, tab_ood |
| 训练协议与稳定性 | run_02–04 日志 | budget_check, variance_decomposition | R-RUNS, B-STEPS-M, STEPS-RUN-M, T-RUN-MIN, SPS, BEST-STEPS-M, VAL-RANGE, ICC-SEED, ICC-INST, tab_budget, F5 |
| 全局检验 | run_07 | friedman_nemenyi | FRIEDMAN-P, CD, N-FAMILY, F6 |
| 算例设计 | run_01 | index | N-GRID, N-VAL, N-SMALL, N-OOD, tab_instances |
| 决策时延 | run_05 | eval_results.decision_time_ms | DT-COH, DT-SPT |

## 6. 学习基线协议

三个基线与主方法共用环境、候选暴露、奖励、编码器结构、交互预算（`training.total_steps`）、验证与
checkpoint 规则，且都能表达保留产能：

| 基线 | 动作 | 学习 |
|---|---|---|
| RuleSel-PPO（DRLG 风格） | 6 条规则 ∪ {hold}，规则再定三元组 | 同一 PPO 学习器（无辅助头） |
| Triplet-DQN（AHP-DQN 风格） | 逐候选 Q（含 no-op 行）+ 三个优先级代理特征 | Double DQN、目标网络、5 步回报、replay 30 万、每个新转移 0.05 步梯度（minibatch 512）、ε 1.0→0.05（前 20% 预算） |
| Hier-DQN（HSDDQN 风格） | 上层 Q 在规则 ∪ {hold}，下层 Q 在所选规则前 3 个候选内 | 两层 Double DQN，下层次态 argmax 限制在 t+n 时的窗口（SARSA 式近似，论文写明） |

## 7. 预注册判据（先于任何正式结果写定）

- **主判据**（grid 档 36 算例，逐算例取 5 个 run 的均值，配对 Wilcoxon，Holm 族 = 全部池化对比）：
  CoH 对 `SPT-Idle*`、对 `OracleRule`、对 `CoH-NoHold`、对最强学习基线，各自 Holm p < 0.05 **且**
  Cliff's δ ≥ 0.33。任何一条不成立都如实报告，不追加机制。
- **分层**：按 DDT 三档与负荷两档分别报告（族内 Holm）；异质性用置换检验；"同时优于全部规则"用交–并
  检验。结论只在成立的档内陈述。
- **消融**：五个消融变体的差值与 Holm p 无论方向都报告；"无增益"照写。
- **精确参照**：small 档报告离线最优、滚动重优化、CoH、非延迟策略与最强规则的均值与差距；CoH 对滚动
  重优化的胜场与 p。
- **效应量口径**：Cliff's δ 用非配对定义；胜场、均值差与 BCa 区间作描述。BCa 固定种子。

## 8. 必须报告的负面结果

- 任一档内 CoH 不优于最强规则或非延迟策略的事实；
- 任一消融不劣于主方法的事实；
- `SPT-Idle*` 与 `OracleRule` 是逐算例事后选择，是规则族的上界而非可部署的规则；
- 承诺评估器的校准误差（ECE）与 Brier；
- 训练稳定性：末 10 次验证极差、崩塌次数、最佳 checkpoint 出现的步数分布。
