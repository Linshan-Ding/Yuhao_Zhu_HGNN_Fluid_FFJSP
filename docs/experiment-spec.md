# 实验设计规格（Experiment Spec）— FSHGRL / DFFSP-HFOI

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-08-28 | 由 opt-paper-codegen 单独使用模式（分支 C，阶段 E）生成。契约来源为重构后的稿件 `cas-sc-template.tex`（分支 `claude/latex-paper-restructure-nfwghr`）及其占位符清单，而非自由格式方案。 |
| v1.1 | 2026-08-29 | 算法优化轮。诊断发现三层结构性问题并据此修订规格，详见下方 §7 变更记录与 §8 预注册判定规则。**本版之前产出的全部实验数据作废。** |

> 本文件是代码仓库的任务书。**论文里每一个 `\PH{}` 与每一张表的 `\dc` 单元格，都必须能在 §6C 映射表里找到产出它的脚本与 CSV 列**；反之每一列落盘都必须有论文去向。映射表无天窗 ⇔ 工程无天窗。

---

## 0. 范式判定

**默认范式**：构造式 DRL（PPO / Actor–Critic + 异构图注意力编码器 + 动作级自注意力）。
不触发模块替换协议——`agent/` 保留 PPO 结构，但按稿件 §4.8.2 引入**行为策略修正**（存 `log b_k` 而非 `log π_old`）。

**环境不可张量化**：本问题是**离散事件仿真**（随机到达、机器完工事件、订单丢弃），且 `step` 内需调用 Gurobi 求解流体 LP。按 codegen 技能"环境并行判定表"，此形态属于**值得引入多进程**一类。当前实现为单进程 + LP 缓存；worker 并行留作可选项，判定与实测加速比记录在 README §2。

---

## 1. 问题定义（与稿件 §3 一一对应）

- **对象**：动态柔性流水车间，高频插单（DFFSP-HFOI）。机器集 $\mathcal{M}$，产品类型 $\mathcal{R}$，每型固定阶段链 $1..J_r$，每阶段多台可选机器。
- **决策**：在每个决策时点，把某个就绪订单的当前工序指派给某台合格空闲机器，动作为三元组 $(o_{rj}, m, s)$；订单维度以**紧急度排序的 top-$K$ 槽**表示（$K=5$）。
- **目标**：最大化订单按时达成率 $\eta = N_c/|\mathcal{S}|$；订单在剩余路径最小加工时间已超过交期时被**丢弃**，丢弃率 $\nu = N_d/|\mathcal{S}|$。
- **奖励**（稿件 Eq. 46）：$r_t = (\Delta N_c - \kappa_d \Delta N_d)/|\mathcal{S}|$，$\kappa_d = 1$。
  **恒等式**：$\gamma=1$ 时 $\sum_t r_t = \eta - \kappa_d\nu$ —— 由 `scripts/run_00_smoke.py` 用随机策略 rollout 强制校验，不通过即报错退出。
- **可选稠密信号**：势函数塑形 $F_t = \gamma\Psi(\omega_{t+1}) - \Psi(\omega_t)$，$\Psi = \min\{\Phi^*, 1\}$，策略不变（Ng et al. 1999）。

---

## 2. 流体松弛（稿件 §3.3）— 相对旧实现的三处规格级改动

| # | 旧实现（`agent/training_simulator.py::_solve_cached_lp`） | 新规格 | 依据 |
|---|---|---|---|
| F1 | 约束为 `service >= min_rate * demand`，**不含交期** | 约束改为 $\hat\delta_{rj}\sum_m \mu_{rjm}u_{mrj} \ge \Phi W_{rj}$ | 稿件 Eq. (21)，回应 R3.2/R4.1 |
| F2 | 无阶段间流平衡约束 | 增加 $W_{rj}\sum_m\mu_{r,j-1,m}u_{m,r,j-1} \ge W_{r,j-1}\sum_m\mu_{rjm}u_{mrj}$ | 稿件 Eq. (24) |
| F3 | 只返回 `(allocations, rates)` | 另返回 $\Phi^*(t)$、基本解支撑集大小、求解/缓存计数 | 势函数塑形 + Prop 1(d) 实证 |

**有效松弛期**：$\hat\delta_{rj}(t) = \max\{\delta_{rj}(t) - \sum_{j'>j}\min_m p_{rj'm},\ \delta_{\min}\}$。

---

## 3. 剪枝与安全网（稿件 §4.5）

$$\mathcal{A}^f_t = \{(o_{rj},m,s)\in\mathcal{A}^{\text{feas}}_t : u^*_{mrj}>\epsilon_f \ \lor\ s\in\mathcal{S}^{\text{crit}}_{rj}(t)\}$$

$\mathcal{S}^{\text{crit}}_{rj}(t) = \{s : d_s - t \le \theta_{\text{crit}}\underline{P}_{rj}(t)\}$，回退规则保证非空（Prop 2(a)）。

$\epsilon_f = 10^{-5}$，$\theta_{\text{crit}}$ 见 `configs/env.yaml`。

---

## 6A. 算例设计

### 6A-1 算例参数表（训练随机构造的分布）

| 参数 | 取值 |
|---|---|
| 每阶段机器数 | 5 |
| 阶段数 $J_r$ | 5 |
| 产品类型数 $R$ | 5 |
| 加工时间 $p_{rjm}$ (s) | $\mathrm{rand}[25,450]$ |
| 订单数 $S$ | $[20,200]$ |
| 到达间隔 $\Delta t$ (s) | $\mathrm{rand}[1,200]$ |
| 交期宽松度 DDT | $\mathrm{rand}[300,2500]$ |

**派生负荷指标**（每个算例族都必须落盘）：$\Lambda = 1/\mathbb{E}[\Delta t]$，$\bar W = \sum_r \pi_r\sum_j \bar p_{rj}$，$\rho_{\text{sys}} = \Lambda\bar W/|\mathcal{M}|$，$\iota = \Lambda\bar p$。

### 6A-2 算例设计表（评测算例，k=1，每单元一个算例）

| 档位 | 数量 | 规模因子 | 结构因子 | 用途 | 论文去向 |
|---|---|---|---|---|---|
| `small` | 16 | $S\in\{6,8,10,12\}$，$R=2$，$J_r=3$，$\|\mathcal{M}\|=6$ | DDT 低/高 2 水平 | 可精确求解，校准真实 gap | Table T-NEW-5 |
| `main` | 15 | $S\in\{50,100,150\}$ | DDT $\in\{500,600,900,1200,1600\}$ 全因子 | 主对比、消融、统计检验 | Tables T-NEW-3/4、PDR、DRL、Wilcoxon |
| `arrival` | 15 | $S=100$ 固定 | $\mathbb{E}[\Delta t]\in\{200,100,50,25,12.5\}$ × 到达过程 {确定性, Poisson, MMPP 突发} | 到达强度扫描 | Table T-NEW-8(a) |
| `ood` | 5 | $S\in\{300,500\}$；$\|\mathcal{M}_{\text{stage}}\|\in\{3,8\}$；$R=8,J_r=7$ | 留在参数表内水平 | 分布外泛化（不重训） | Table T-NEW-8(b) |
| `val` | 5 | 按参数表随机 | — | 选 best checkpoint，不进论文 | 训练曲线 |
| `case3d` | 9 | $S\in\{50,100,150\}$ × DDT $\in\{600,900,1200\}$ | 6 阶段 3D 打印工艺链 | 情景可迁移性 | Table 案例研究 |

**防混淆**：`ood` 档只推规模/结构因子，不同时改到达分布；到达分布偏移单独由 `arrival` 档承担。

---

## 6B. 数据落盘清单（唯一真源，全部 CSV）

| 文件 | 关键列 | 产出脚本 |
|---|---|---|
| `data/instances/<tier>/instance_*.csv` + `index.csv` | `instance_id,tier,S,R,J,M,DDT,arrival_process,E_dt,Lambda,W_bar,rho_sys,iota` | `run_01` |
| `result/<run>/log.csv` | `iter,steps,eta_val,reward,policy_loss,value_loss,entropy,approx_kl,clip_frac,ratio_max,ratio_bound,sps,gpu_mem_gb,fluid_solve_count,fluid_cache_hit,phi_star_mean` | `run_02/03/04` |
| `result/<run>/config_snapshot.yaml`、`commit.txt` | 生效配置 + git commit | `run_02/03/04` |
| `result/eval_results.csv` | `instance_id,tier,method,variant,run_id,eta,nu,decision_time_ms,steps,feasible` | `run_05` |
| `result/pruning_stats.csv` | `instance_id,A_feas_mean,A_feas_max,A_f_mean,A_f_max,prune_ratio,p_singleton,fallback_rate,retention_all,retention_crit,retention_se,delta_eta,t_lp_ms,t_enc_ms,t_pol_ms,zeta,support_size` | `run_06` |
| `result/pruning_sensitivity.csv` | `eps_f,prune_ratio,retention_all,retention_crit,eta,decision_time_ms` | `run_06` |
| `result/exact_results.csv` | `instance_id,S,eta_off_gurobi,eta_off_cpsat,solver_time_s,eta_online_exact,eta_fshgrl,eta_best_pdr,eta_best_drl,abs_gap,rel_gap,replay_match` | `run_07` |
| `result/arrival_results.csv` | `E_dt,rho_sys,iota,arrival_process,eta,nu,phi_star_mean,decision_time_ms` | `run_08` |
| `result/ood_results.csv` | `condition,method,eta,eta_matched,retention` | `run_08` |
| `result/shift_matrix.csv` | `shift_axis,train_cond,test_cond,eta,retention` | `run_08` |
| `result/reward_exploration.csv` | `panel,config,eta,eta_ci_lo,eta_ci_hi,nu,steps_to_90pct,ratio_max,ratio_bound,approx_kl` | `run_09` |
| `result/case3d_results.csv` | `case,DDT,S,eta_best,eta_avg,ci_lo,ci_hi,decision_time_s,eta_best_rule,eta_avg_rule,eta_best_drl,imp_pct,gap_pct` | `run_10` |
| `result/stats_summary.csv` | `comparison,R_plus,R_minus,p_raw,p_holm,p_bh,r_rb,cliff_delta,A12,lmm_est,lmm_ci_lo,lmm_ci_hi` | `run_11` |
| `result/variance_decomposition.csv` | `source,var_component,icc` | `run_11` |
| `result/friedman_nemenyi.csv` | `method,mean_rank,cd,friedman_stat,friedman_df,friedman_p` | `run_11` |
| `result/figures/*.pdf` | F-NEW-2 / F-NEW-4 / F-NEW-5 + 消融面板 + 训练曲线面板 | `run_12` |
| `result/paper_values.tex` | 全部 `\PH{}` 宏定义 + 已填数的 LaTeX 表格 | `run_13` |

---

## 6C. claim → 实验 → 数据 → 论文占位符 映射（核心，无天窗）

| # | claim / 论文需求 | 实验 | 落盘 | 论文占位符 |
|---|---|---|---|---|
| C1 | 流体松弛是有效松弛且与目标对齐 | 精确解对照 + 回放校验 | `exact_results.csv` | `A3`、`P-MILPCHK`、Table T-NEW-5 全部 `\dc` |
| C2 | 剪枝安全、紧凑、保留临界动作 | 剪枝统计 + oracle 保留率 + $\epsilon_f$ 扫描 | `pruning_stats.csv`、`pruning_sensitivity.csv` | `A1`、`A2`、`O1`、`O2`、`P-TIE`、`S2`–`S6`、Table T-NEW-4、Fig F-NEW-2 |
| C3 | 宏观引导的贡献 ≠ 动作空间缩减 | 9 变体因子化消融（含同基数随机/启发式对照） | `eval_results.csv` → `stats_summary.csv` | `S1`、Table T-NEW-3、Fig 消融面板(b) |
| C4 | 奖励到达不变 + 行为策略修正有效 | 奖励/探索消融 3 面板 | `reward_exploration.csv` | `H4`、Table T-NEW-9 |
| C5 | 优于规则与学习基线（等预算） | 主评测 + 基线适配 | `eval_results.csv` | `A4`、`A5`、`A6`、`B1`、`B2`、`H5`、Table T-NEW-6、PDR/DRL 表 |
| C6 | 高频插单可度量、越过饱和仍有效 | 到达强度扫描 + 过程形态 | `arrival_results.csv` | `L0`–`L4`、Table T-NEW-8(a) |
| C7 | 分布外泛化（不重训） | OOD 档 + 偏移矩阵 | `ood_results.csv`、`shift_matrix.csv` | `S7`、`S8`、Table T-NEW-8(b)、Fig F-NEW-4 |
| C8 | 统计结论稳健 | BCa CI + 效应量 + 多重校正 + Friedman/Nemenyi + LMM/ICC | `stats_summary.csv`、`variance_decomposition.csv`、`friedman_nemenyi.csv` | `R1`–`R5`、Table Wilcoxon、Fig F-NEW-5 |
| C9 | 情景可迁移性（非工业验证） | 3D 打印 9 算例 | `case3d_results.csv` | 案例研究表全部 `\dc` |
| C10 | 复现性协议完整 | 各 run 落盘配置/commit | `config_snapshot.yaml`、`commit.txt` | `R4`、`H1`–`H3` |

**天窗检查**：论文 35 个唯一 `\PH` id 与 10 张表的 `\dc` 单元格，均已在上表出现；`run_13_fill_placeholders.py` 会在缺任何一项时报错列出，即自动化的天窗检查。


---

## 7. v1.1 变更记录（算法优化轮）

每条都注明"改了什么、为什么、影响哪个下游"。规格级改动一律登记，不允许静默偏离。

### 7.1 修复：策略从未学习（代码缺陷，非规格问题）

`agent/networks.py` 的动作级自注意力写成 `feats = attn @ v`，丢失残差与 LayerNorm。
候选集仅 1–5 个元素且 68 维中 65 维在候选间相同 ⟹ 注意力权重近似均匀 ⟹ 所有候选被映射为
`mean(v)` ⟹ logits 全等 ⟹ 策略**恰好**均匀。实测 `‖∂log π/∂θ_actor‖` = 1.77e-05（无注意力时
7.98e-02，相差 4501 倍），归一化熵恒为 1.000000。

连带失效：`b_k=(1-ε)π+ε/n` 退化为 π ⟹ `ratio_max≡1.000`、`clip_frac≡0`、`approx_kl≡0`，
log.csv 的三个诊断量全部丧失诊断能力；贪心评测退化为"取枚举顺序第 0 个"。

> 该缺陷由本仓库重建时引入。原作者 `HGHH_model.py:520` 写的是
> `norm_layer(candidate_features + attended)`，残差与 LayerNorm 都在。

修复后实测：`‖grad‖` = 2.00e+00，熵 0.9988（不再全等），三个诊断量恢复变化
（`ratio_max` 1.14–1.73、`clip_frac` 0–0.020）。**影响下游**：v1.1 之前所有训练结果作废。

### 7.2 规格改动一览

| # | 改动 | 依据（实测） | 影响 |
|---|---|---|---|
| 7.2.1 | `rollout_episodes` 2→8，`minibatch_size` 512→128 | 原配置下 minibatch(512) > buffer(~434)，退化为全批量，1000 epoch 仅 3000–9000 个梯度步 | 训练协议 |
| 7.2.2 | `epsilon0` 1.0→0.3，`anneal_epochs` 1000→350 | ε=1 时 `b_k` 恒为均匀分布，PPO 裁剪区间等价于"强制 π 保持均匀"，是持续的反向拉力；原设定全程平均 ε=0.5 | 训练协议 |
| 7.2.3 | `approx_kl` 基准从 `log b_k` 改为 `log π_old` | 前者衡量的是行为分布与目标策略的距离，在第一次梯度步前就非零，会把 `update_epochs` 误削成 1 | 训练协议 |
| 7.2.4 | 观测全面无量纲化 + 网络加 LayerNorm | `act_feat[:,1]`（唯一有区分度的列）量纲 0.007，其余特征 O(1)，相差四个数量级 | 6B `log.csv` 列不变 |
| 7.2.5 | 动作特征增加**临界比** `slack / 剩余路径最小加工时间`，ACT_DIM 3→4 | 交期目标下判别力最强的单一信号，原观测完全没有 | 6B 不变 |
| 7.2.6 | critic 增加全局状态：未到达比例 / 未结清比例 / 时间进度 / 丢弃率 | 原 critic 输入 33 维且**看不到未到达订单**，价值函数结构不可辨识 | 6B 不变 |
| 7.2.7 | 流体目标从 max-min 交期可行比改为**吞吐对齐** `max Σ min(δ̂λ, W)` | max-min 是均衡型目标，与"按时完工件数"在过载下要求的选择性放弃方向相反 | §2 F1 改写；Φ* 语义变为"可按时交付工作量占比"，天然落在 [0,1] |
| 7.2.8 | `theta_crit` 1.0→1.5 | θ=1.0 时安全网是死分支，实测生效率 0/747，Prop 2(c) 空洞成立 | §3 |
| 7.2.9 | 新增**规则行为克隆热启动**（`training.bc_warmup_steps=3000`，专家为 SPT） | PPO 随机初始化时性能远低于最强规则；热启动给出可证的性能下界 | 方法节须如实报告；新增消融 `nobc.yaml` |
| 7.2.10 | 新增消融变体 `fluid_maxmin.yaml`、`nobc.yaml` | 对应 7.2.7 与 7.2.9 | 6C 消融矩阵扩到 11 个变体 |

### 7.2b Phase 0 验收结果（2026-08-29，main 档 15 算例，**不改算例、不改 MDP**）

修复后用一次截断的探针训练（60 epoch 请求，实际跑到约 18 epoch，best checkpoint 在
epoch 10）贪心评测，全部方法走同一条 `eval.py` 路径、同一批算例：

| 方法 | mean η | 与 FSHGRL 之差 | p_raw | p_holm | Cliff δ | Â₁₂ |
|---|---|---|---|---|---|---|
| **FSHGRL（探针）** | **0.7911** | — | — | — | — | — |
| SPT | 0.7964 | −0.0053 | 0.237 | 0.237 | 0.004 | 0.502 |
| RRC | 0.5393 | +0.2518 | 6.1e-05 | 4.3e-04 | 0.400 | 0.700 |
| Random | 0.5207 | +0.2704 | 6.1e-05 | 4.3e-04 | 0.444 | 0.722 |
| MOR | 0.5160 | +0.2751 | 9.8e-04 | 4.9e-03 | 0.413 | 0.707 |
| FIFO | 0.5160 | +0.2751 | 9.8e-04 | 4.9e-03 | 0.413 | 0.707 |
| EDD | 0.5160 | +0.2751 | 9.8e-04 | 4.9e-03 | 0.413 | 0.707 |
| MWKR | 0.5151 | +0.2760 | 1.5e-03 | 4.9e-03 | 0.400 | 0.700 |

**验收门（须显著优于 Random）：通过**（+0.2704，p_holm = 4.3e-04，Cliff δ = 0.444）。
修复前 FSHGRL 与 Random 不可区分；诊断量 `ratio_max ≡ 1.000`、`clip_frac ≡ 0`、
`approx_kl ≡ 0`，修复后分别恢复到 1.14–1.73、0.9%–4.2%、~2e-03，
`‖∂log π/∂θ_actor‖` 从 1.77e-05 回到 2.0e+00 量级。

两条必须如实记录的观察：

1. **MOR / FIFO / EDD 的 η 完全相同（0.5160，逐算例相等）**，MWKR 仅差 0.0009。
   直接证据说明 §7.3.1–7.3.2 的两处退化是真的：常数 DDT 使 EDD≡FIFO，
   而 96% 的单订单决策点使这些不含机器信息的规则退化为同一策略。
2. **FSHGRL 与 SPT 统计上不可区分**（δ=0.004），且逐算例看在 8/15 上**完全相等**、
   5 个略负、2 个略正。这不是训练不足，而是当前算例设计下**可学的最优策略本身就
   近似等于 SPT**——多候选时点里 100% 只能选机器，选最快的机器就是 SPT。
   这正是 Phase 1 必须重做算例的理由：在现有设计上继续调参没有可争取的余量。

### 7.3 待执行的规格改动（Phase 1–2）

| # | 改动 | 依据 |
|---|---|---|
| 7.3.1 | 每单独立抽交期 `due = arrival + DDT·U[0.7,1.4]` | 现为 `due = arrival + 常数 DDT`，实测每算例 `due−arrival` 唯一值=1 ⟹ **EDD ≡ FIFO ≡ 环境自带排序键**，订单维度永无信息 |
| 7.3.2 | 主档负荷从 ρ_sys≈0.47 提到 ≈1.2 | 实测订单维度激活率：ρ=0.47→6%，ρ=0.97→37%，ρ=1.83→82%。当前主档 96% 决策点只有 1 张订单可选、84% 只有 1 种工序类型 |
| 7.3.3 | 剔除饱和档 DDT=1600 | 全部方法在该档均为 1.000，无区分度 |
| 7.3.4 | 增加 no-op（主动空闲）动作 | 当前是严格 non-delay 调度，而 non-delay 调度类不含最优解；规则基线按构造都是 non-delay，这是学习策略可差异化的能力 |

---

## 8. 预注册判定规则（在跑主实验**之前**确定，避免事后择优）

- **主指标**：main 档 15 个算例上的订单按时达成率 η，实例级配对 Wilcoxon 符号秩检验，
  Holm–Bonferroni 校正后 α = 0.05。
- **主要对照**：FSHGRL vs **最强单条调度规则**（当前实测为 SPT）。次要对照：全部消融变体与三个学习基线。
- **成功判据**：校正后 p < 0.05 **且** Cliff's δ ≥ 0.33（中等以上效应量）。仅有显著性而效应量微弱不算成立。
- **核心机制判据**（C3）：FSHGRL vs FSHGRL-RP（同基数随机剪枝）的差值，是"宏观引导"区别于
  "单纯缩小动作空间"的唯一证据。当前以固定 SPT 策略实测该差值为 **+0.195**
  （流体剪枝 η=0.4617 vs 同基数随机 η=0.2667，二者 |A_f| 分别为 2.61 与 2.79）。
- **若主判据不成立**：不再追加机制，改为**收窄论断到成立的区间**——按负荷分层报告，
  明确写出"在 ρ≥1 的拥塞区与紧交期档优于全部基线；在轻载区与 SPT 相当"，
  并如实报告 SPT 的强势。带着不成立的 claim 写论文是过度承诺。
- **必须报告的负面结果**：剪枝相对不剪枝的解质量代价（当前实测 −0.045），
  以及流体解作为**优先级信号**时劣于 SPT（−0.083）——它的价值在于**保留哪些动作**，
  而不在于给动作打分。
