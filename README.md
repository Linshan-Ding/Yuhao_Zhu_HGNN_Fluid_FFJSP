# FSHGRL — 高频插单动态柔性流水车间调度

本仓库是两篇论文的实验工程：已录用的 *Fluid-Guided Sparse Heterogeneous Graph Reinforcement
Learning for Real-Time Scheduling in Dynamic Flexible Flow Shops with High-Frequency Order
Insertion*（FSHGRL，§1–§10 的流水线，实验已全部完成），以及正在做的新论文（Commit-or-Hold，
§0 的试点）。

**本 README 不是项目简介，而是复现手册**：把 §3–§8 的命令整段复制进终端顺序执行，
即可产出论文所需的**全部**实验数据。最后一步 `run_13` 会把结果写成
`result/paper_values.tex`，并在任何一个论文占位符缺数据时**报错列出缺口**——
论文里开天窗的表格会在这里被拦下。

---

## 0. 进度与下一步：新论文的试点（Commit-or-Hold）

FSHGRL 的实验矩阵（15 个方法 × 5 run × 250 epoch、`run_05`/`run_06`/`run_08`）已经在 `master`
上完成，原稿已录用，`run_07`/`run_09`/`run_10` 不再需要为它运行。接下来的工作是新论文
**"学会保留产能：超负荷柔性流水车间里非延迟调度的代价"** 的试点。它的证据来自 FSHGRL 的全预算
结果：学会派工但不能等待的策略 ≈ SPT，允许等待后每个交期水平都赢最强规则，而流体、图注意力等
组件对学习策略没有可测增益（`docs/experiment-spec.md` §9 与计划记录）。

试点只回答一个问题：在最小骨干（无流体、MLP 编码、无自注意力、无 BC、无塑形，保留 no-op）上，
**承诺评估器**（P1）和**显式等待门控**（P2）各自是否值得进最终方法。采纳规则在试点开跑前已写进
`docs/experiment-spec.md` §9.3，`scripts/_pilot_report.py` 只做机械核对。

| 配置 | 文件 | 内容 |
|---|---|---|
| P0 | `configs/coh/p0_backbone.yaml` | 最小骨干；候选按"可救优先"暴露（规则评测用同一配置） |
| P1 | `configs/coh/p1_critic.yaml` | P0 + 承诺评估器 p̂(o|s)：候选现在派出后按时完成的概率，实际结果做监督 |
| P2 | `configs/coh/p2_gate.yaml` | P1 + 等待门控 P(hold) = σ(β(logit ĥ − logit max p̂) + c(s)) + 等待前景评估器 ĥ |
| P3 | `configs/coh/p3_onpolicy.yaml` | 第一波胜者 + 纯 on-policy（关掉 ε-greedy 行为混合），第二波再跑 |

### 0.1 合并本分支、跑冒烟

```
git fetch origin
git checkout master
git merge origin/claude/latex-paper-restructure-nfwghr
python scripts/run_00_smoke.py
```

冒烟约 8 分钟。第 4–5 步是新增的 CoH 检查：用 p2 配置训 2 个 epoch，然后核对旧 checkpoint 在新
代码下的评测逐位不变、联合分布归一、监督标签回填完整。任何一项不过都会报错退出。

### 0.2 第一波：P0 / P1 / P2 各 3 个 run

每个 run 250 epoch。没有流体 LP 之后每个 epoch 快得多：4 核容器里单线程实测 p0 约 135 步/s
（随机初始化的前几个 epoch 18–20 s，策略学会派工后每个 epoch 的决策数会增加，估 25–35 s），
p2 约 90 步/s（24–40 s/epoch，含每 10 个 epoch 一次的验证）。按此估算 250 epoch 一个 run
p0 约 1.5–2.5 小时、p2 约 2.5–3.5 小时；跑完第一个 epoch 后用 `log.csv` 的 `elapsed_s` 按你的
机器重新估。bash：

```
python scripts/run_14_pilot_coh.py --only p0 &
python scripts/run_14_pilot_coh.py --only p1 &
python scripts/run_14_pilot_coh.py --only p2 &
wait
python scripts/_pilot_report.py
```

PowerShell（PyCharm 内置终端）开三个标签页各执行一行，都结束后再跑报告：

```
python scripts/run_14_pilot_coh.py --only p0
python scripts/run_14_pilot_coh.py --only p1
python scripts/run_14_pilot_coh.py --only p2
python scripts/_pilot_report.py
```

- 每个训练进程默认单线程（脚本里设了 `OMP_NUM_THREADS=1`）：网络很小，多线程没有收益，
  而几个进程各开满线程会互相自旋等待，实测慢一个数量级。三个配置并行需要 3 个空闲核。
- 一个 run 只有 `log.csv` 记满 250 行才算完成，中断的会从头重训；已完成的会跳过。
- `_pilot_report.py` 把每个 run 的 `checkpoint_best` 在 15 个测试算例上贪心评测一次（可续跑，
  结果在 `result/pilot_eval.csv`），八条规则在同一环境设置下重评（`result/pilot_rules.csv`），
  然后打印 §9.3 的裁决表并写 `result/pilot_summary.csv`。裁决表里的"采纳 / 不采纳"就是结论，
  不要按结果回头改规则。

### 0.3 第二波：P3

第一波胜者若不是 P2，先把 `configs/coh/p3_onpolicy.yaml` 里 `coh:` 一段改成胜者的取值，然后：

```
python scripts/run_14_pilot_coh.py --only p3
python scripts/_pilot_report.py
```

### 0.4 试点之后

最终配置、工况网格算例、完整对比与消融矩阵、学习基线修复、统计与论文表格按
`docs/experiment-spec.md` §9.4 进行，届时更新本节。试点 run（`coh_p*_run*`）只用于选配置，
不进最终矩阵；它们被 `scripts/_bootstrap.py` 的排除前缀挡在 `run_05`/`run_11` 之外。

FSHGRL 的结果文件（`eval_results.csv` 等）保持不动；`run_07` 的精确解与在线重优化会在新论文的
small 档参照里复用，现在不必跑。

---

## 1. 问题假设

- **问题**：动态柔性流水车间，高频插单（DFFSP-HFOI）。产品类型 $r$ 走固定阶段链
  $1..J_r$，每阶段有多台可选机器，机器异构（合格机器集与加工时间随工序类型变化）。
- **动态性**：订单按到达过程随机到达，到达时刻才可见。剩余路径最小加工时间已超过交期的
  订单被**丢弃**，释放产能给仍可交付的订单。
- **决策**：每个决策时点（有空闲机器且有就绪工序）指派一个三元组
  $(o_{rj}, m, s)$；订单维度用**按紧急度排序的 top-$K$ 槽**表示（$K=5$）。
- **目标**：最大化按时达成率 $\eta = N_c/|\mathcal{S}|$。
- **奖励**：$r_t = (\Delta N_c - \kappa_d \Delta N_d)/|\mathcal{S}|$，$\kappa_d = 1$。
  它**对订单到达不变**——到达既不改变 $\Delta N_c$ 也不改变 $\Delta N_d$，
  因此不会把环境事件记到动作头上。$\gamma = 1$ 时恒有
  $\sum_t r_t = \eta - \kappa_d\nu$，此恒等式由 §3 的冒烟脚本强制校验。
- **流体松弛**：交期感知的线性规划，目标是**交期可行比**
  $\min_{rj} \hat\delta_{rj}\lambda_{rj}/W_{rj}$；$\Phi^* \ge 1$ 是活跃订单集全部按时
  交付的必要条件。解同时用于状态特征、动作剪枝与势函数塑形。
- **剪枝**：保留流体分配 $u^* > \epsilon_f$ 的机器–工序对，**外加**对临界订单
  （$d_s - t \le \theta_{\mathrm{crit}}\underline{P}$）无条件开放全部合格空闲机器；
  两者都为空时回退到全可行集，保证永不死锁。

契约文件见 `docs/experiment-spec.md`（算例设计 6A、落盘清单 6B、claim→实验→数据映射 6C）。
偏离规格前先在该文件的变更记录里登记。

---

## 2. 环境配置

Python 3.11+。**不需要任何商业求解器授权**：

- 流体 LP 只用 SciPy 自带的 HiGHS（`linprog(method="highs")`）。LP 退化时最优解不唯一，
  剪枝读的正是解的支撑集，所以训练与评测必须用同一个求解器；旧版在 max-min 模式下会先试
  Gurobi、授权失效才回退 HiGHS，同一份代码在两台机器上可能剪出不同的动作集，现已删除。
- 精确解由两个求解器独立给出：OR-Tools CP-SAT，以及用 SciPy 的 `milp`（HiGHS 分支定界）
  直接求解稿件 Eqs. (1)–(11) 的 MILP。CP-SAT 在一个保序的整数时间尺度上求解：工时为整数时，
  左对齐排程的每个时刻都形如"某订单到达时刻 + 整数"，按到达时刻小数部分的名次编码后，
  到达、前后序、同机不重叠与交期四类比较全部保持，CP-SAT 的最优值因此就是连续时间模型
  的最优值。旧版把到达向上取整、交期向下取整，实测在 `small_S20_DDT600_c1` 上把 eta_off
  从 0.60 低估成 0.55。

```
pip install -r requirements.txt
```

各阶段实测代价（单机 CPU，无 GPU；本工程的环境是离散事件仿真，rollout 在 CPU 上推进）：

| 阶段 | 脚本 | 预计耗时 |
|---|---|---|
| 冒烟自检 | `run_00` | 约 3 分钟 |
| 算例准备 | `run_01` | < 1 分钟 |
| 主方法训练（5 run × 1000 epoch） | `run_02` | 约 40–60 小时 |
| 消融训练（8 变体 × 5 run） | `run_03` | 约 250–350 小时 |
| 基线训练（3 方法 × 5 run） | `run_04` | 约 60–90 小时 |
| 主评测 | `run_05` | 约 4–6 小时 |
| 剪枝分析（含前瞻 oracle） | `run_06` | 约 6–10 小时 |
| 精确解 + 回放 + 在线重优化 | `run_07` | 约 2–6 小时（`--jobs 4`，主要是 MILP，见 §8） |
| 到达强度 + OOD | `run_08` | 约 2–4 小时 |
| 奖励/探索消融 | `run_09` | 约 20–30 小时 |
| 案例研究 | `run_10` | 约 2–3 小时 |
| 统计聚合 / 绘图 / 回填 | `run_11`–`run_13` | 约 10 分钟 |

训练是最大开销，且与 `configs/algo.yaml` 的 `total_epochs` 线性相关。想先看趋势，
把该值调小再跑；**改参数改 configs，跑实验只复制命令**。

**关于并行**：本环境是离散事件仿真（随机到达、机器完工事件、订单丢弃），且 `step`
内要解流体 LP，属于难以张量化的一类，因此没有采用 GPU 向量化环境。当前实现是
单进程 + 流体 LP 的事件触发缓存（缓存未命中率 $\zeta$ 落在 `pruning_stats.csv`）。
多进程 worker 并行是可选的下一步，引入前须实测加速比 ≥ 1.5× 才值得其复杂度。

---

## 3. 冒烟自检（先跑这条，分钟级）

微型规模走通"算例 → 奖励恒等式 → 训练 → 评测 → 精确解 → 统计"整条链路。
投入完整算力前先确认环境无误。

```
python scripts/run_00_smoke.py
```

其中第 2 步会用随机策略强制校验 $\sum_t r_t = \eta - \kappa_d\nu$，
第 5 步会校验精确解回放进仿真器后完工时间一致。**任何一项不通过即报错退出**——
这两条是"环境建模正确"的硬性证据，不通过就不要继续往下投算力。

---

## 4. 数据准备（生成六档固定算例）

```
python scripts/run_01_prepare_data.py
```

产物：`data/instances/{small,main,arrival,ood,val,case3d}/*.csv` + `index.csv`。

评测算例一次生成、永久固定、随论文发布——**复现基准是这些算例文件本身加多次独立
run，不是随机种子**（本工程不设随机种子）。训练算例不预生成，每个训练周期按算例
参数表现场随机构造。`index.csv` 逐算例记录负荷指标 $\Lambda$、$\bar W$、
$\rho_{\mathrm{sys}}$、$\iota$ 与体制标签，论文中"高频插单"的定义直接取自这几列。

已存在的档位会自动跳过；要重建先删 `data/instances/`。

### 算例设计的三处取值是标定出来的，不要随手改

`configs/instance.yaml` 里有三个参数直接决定"实验能不能分辨出方法优劣"。它们的取值
经扫描确定，改动前请先重跑对应的判据（`docs/experiment-spec.md` §7.4–7.5 有完整数据）：

| 参数 | 取值 | 改动后会发生什么 |
|---|---|---|
| `param_table.ddt_spread` | `[0.7, 1.4]` | 若退回常数 DDT，则每张订单的 `due − arrival` 相同，**EDD ≡ FIFO ≡ 环境候选集排序键**，订单维度不携带信息。实测此时 MOR/FIFO/EDD 在 15 个算例上逐位相等 |
| `design.main.target_rho_sys` | `1.6` | 决定三个决策维度的激活率。实测多候选时点中"能选订单/工序类型/机器"的比例：ρ=1.2 → 33%/32%/85%，ρ=1.6 → 54%/51%/70%，ρ=2.0 → 68%/66%/57%。ρ 过低则问题退化为选机器，过高则机器几乎不同时空闲 |
| `design.main.ddt_levels` | `[700,900,1100,1400,1800]` | 须避开饱和档。ρ=1.6 下最强规则 η 依次为 0.40/0.46/0.53/0.80/0.96，而 DDT=2400 时所有方法均为 1.000——饱和档不只是加噪声，它会把任何聚合统计量往"无差异"方向稀释 |

同理，`design.small.ddt_levels = [260, 320]` 也是标定值：加入交期扰动后 0.7×200=140
已低于该档的路径下界（117–254），实测会出现整个算例 6/6 订单到达即 hopeless、
精确解恒为 0 的情况，最优性间隙就没有信息量了。

---

## 5. 训练主方法（5 次独立 run，循环写在脚本里）

```
python scripts/run_02_train_main.py
```

产物：`result/fshgrl_run1..5/{log.csv,checkpoint_best.pt,checkpoint_last.pt,config_snapshot.yaml,commit.txt}`。

训练分两段：先用最强调度规则（SPT）做 `training.bc_warmup_steps` 步行为克隆热启动，
再进 PPO。热启动不是调参技巧而是**性能下界**——随机初始化的策略远不如一条好规则，
PPO 的前若干个 epoch 会全花在追赶规则上。实测热启动把验证档 η 从 0.09（随机初始化）
带到 0.53，而专家 SPT 本身是 0.51，即 PPO 从规则水平起步。日志里会打印这一行：

```
[fshgrl_run1] BC 热启动完成：expert=SPT steps=3000 loss=0.2866 eta_val=0.4900 (33s)
```

**若这行的 `eta_val` 明显低于规则水平，不要继续训练，先查原因。**
一个已经踩过的坑：把 no-op 那一维从交叉熵里屏蔽掉（本意是"别把规则永不空闲的限制
一起克隆过来"），会让 no-op 的 logit 拿不到任何梯度、停在随机初值上，贪心策略于是
随机空转，热启动后 η 只有 0.06。正确做法是让交叉熵在**完整动作集**上归一化。

### 训练诊断：先看这四列，全是常数就说明策略没在学

`log.csv` 除常规指标外还记录 `ratio_max` 与 `ratio_bound`——后者是
$|\mathcal{A}^f_t|/\varepsilon_k$，即修正后重要性比率的理论上界。实测比率越界即说明
行为策略修正被破坏，是训练诊断的第一道关。

| 列 | 健康表现 | 异常含义 |
|---|---|---|
| `entropy` | 随训练下降，且 $< \log\|A_f\|$ | 恒等于 $\log\|A_f\|$ ⟹ 策略恒为均匀分布 |
| `ratio_max` | $> 1$ 且逐 epoch 波动 | 恒等于 1.000 ⟹ $b_k \equiv \pi$，策略没动过 |
| `clip_frac` | 数个百分点 | 恒等于 0 ⟹ 同上 |
| `approx_kl` | $10^{-4}$–$10^{-2}$ 量级 | 恒等于 0 ⟹ 同上 |

**不要把 `eta_train` 当学习曲线看。** 训练算例每个 epoch 现场随机构造，ρ∈[0.8,1.8]、
S∈[20,200]、DDT∈[400,1100]，难度差异极大：把策略固定成 SPT 实测 24 个随机训练算例，
eta 的标准差就有 0.175，8 个算例（= 1 个 epoch）均值的标准误约 **±0.062**。
也就是说策略一动不动时 `eta_train` 也会在 ±0.12 的范围里抖。判断学习进度只看
`eta_val`（固定的验证算例，逐 checkpoint 配对可比）。

这三个诊断量同时为常数是一个**明确的故障信号**而不是"训练很稳定"。本工程修复过一次
这样的缺陷：动作级自注意力写成 `feats = attn @ v` 而漏了残差与 LayerNorm，候选集只有
1–5 个元素且特征高度相似，注意力把所有候选映射成同一向量，logits 全等、softmax 恰好
均匀，`‖∂log π/∂θ‖` 只有 1.77e-05（正常量级 1e-2 ~ 1e0）。修复后同一诊断量恢复为
`ratio_max` 1.14–1.73、`clip_frac` 0.9%–4.2%。

`zeta` 列是流体 LP 的缓存未命中率，对应稿件 §4.9 的摊销系数。健康值约 0.6；若接近
1.000，说明缓存键里混入了与 LP 解无关的分量（本工程曾因把"当前空闲机器集"放进键里
而得到 ζ=0.997，LP 几乎每个决策点重解一次，占 rollout 用时约一半）。

想看实时曲线：**另开一个终端标签页**执行 `python -m visdom.server`，浏览器打开
`http://localhost:8097`。不开也不影响训练。

---

## 6. 基线与消融

```
python scripts/run_03_train_ablations.py
python scripts/run_04_train_baselines.py
```

消融变体定义在 `configs/ablation/*.yaml`，清单固定在脚本顶部常量里，增删变体不改
README 命令。当前共 11 个变体，分两组：

**五机制网格**（每个变体只去掉一个机制）：`noff` 去流体节点特征、`nofp` 去流体剪枝、
`nofa` 去势函数塑形、`nosa` 去动作级自注意力、`nohg` 把图编码器换成等参数量 MLP、
`noall` 同时去掉全部流体成分。其中 **FSHGRL-RP 与 FSHGRL-HP 是同基数对照**：候选集
大小与 FSHGRL 相同但不含流体信息，二者与 FSHGRL 的差距就是"宏观引导的内容"带来的
部分——这正是分离"引导"与"单纯缩小动作空间"的关键设计。

**三个设计决策变体**（不动网格，各改一处，且每一个都有可能推翻对应的论断）：

| 变体 | 改动 | 检验什么 |
|---|---|---|
| `maxmin` | 流体目标退回 max-min 交期可行比 | 均衡型宏观目标与计数型准则是否真的错配。若该变体不劣于主方法，则稿件 §3.3 关于"目标对齐"的论证不成立 |
| `nonoop` | 关闭主动空闲 | 跳出 non-delay 策略类是否真有价值。**若训练后 no-op 使用率≈0，如实记录并把该机制从方法中撤掉**，不要留着一个不起作用的机制 |
| `nobc` | 关闭规则热启动 | 相对 SPT 的增益是学出来的还是从专家那里继承的。这是最容易被审稿人问到的一条 |

**跑消融前必须先确认主方法的训练诊断正常**（见 §5 的四列表）。诊断异常时消融表会
自相矛盾：例如动作注意力塌缩未修复时 `nosa`（去掉注意力）反而会优于主方法，
`onpolicy`（ε₀=0）也会优于主方法——这两个"意外"本身就是主方法有缺陷的信号。

三个学习基线共用同一环境、同一算例、同一奖励与同一交互预算，且都拿不到流体解
（那正是被检验的机制）。它们是**适配版**而非原文逐字复现，适配细节写在
`agent/baselines/drl_baselines.py` 的模块文档里，与论文的协议表一致。

---

## 7. 评测与实验数据生成

```
python scripts/run_05_eval_main.py
python scripts/run_06_pruning_analysis.py
python scripts/run_07_exact_optimality.py
python scripts/run_08_arrival_ood.py
python scripts/run_09_reward_exploration.py
python scripts/run_10_case_study.py
```

评测脚本自动发现 `result/` 下的 checkpoint，不需要手填任何路径。

- `run_06` 除剪枝率外还测 **oracle 动作保留率**：oracle 是一步前瞻（公共随机数、并报
  rollout 标准误——前瞻 oracle 自身是估计量，只报点值会误导）。它同时输出
  `P(|A_f|=1)`，即候选集退化为单元素的比例。
- `run_07` 在小档用 CP-SAT 与 HiGHS MILP 两个求解器求离线最优并交叉核对，把 CP-SAT 的
  最优排程代入字面 MILP 的约束矩阵逐行检查，再让两份排程分别**驱动离散事件环境逐道
  工序回放**，环境自己记下的按时完工数必须与求解器一致——这是"MILP 与仿真器描述同一个
  问题"的直接证据。它还给出在线滚动重优化：每个到达时刻只对已知订单精确重排、不看未来，
  精确但短视，既不是在线策略的上界也不是下界。详见 §0.2。
- `run_09` 的 13 行里 4 行复用主方法的 5 个 run，另 9 个配置各训练 1 个 run，详见 §0.3。
- `run_08` 的三种到达过程按构造共享同一平均到达率，差异只反映突发性；
  OOD 档**不重训**——之所以可行，是因为图规模由 $|\mathcal{O}|\times|\mathcal{M}|$
  固定、剪枝后动作空间的界与订单数无关。

---

## 8. 统计聚合与绘图

```
python scripts/run_11_aggregate_stats.py
python scripts/run_12_make_figures.py
python scripts/run_13_fill_placeholders.py
```

`run_11` 产出三层证据：BCa 自助置信区间（描述）、配对 Wilcoxon + 以实例为随机截距的
混合效应估计（推断）、效应量（$r$、Cliff's $\delta$、$\hat A_{12}$）与
Holm/BH 校正（量级与多重比较），外加方差分解 ICC 与 Friedman/Nemenyi。
**结论从校正后的 p 值与效应量读，不从未校正的 p 值读。**

`run_13` 是闭环的最后一步：把全部 CSV 汇成 `result/paper_values.tex`，
每个论文占位符对应一条 `\newcommand`。缺任何一项即报错列出缺口。

一键跑完 §4–§8：

```
python scripts/run_all.py
```

任一步失败即停。重跑时并不会跳过全部已完成步骤：`run_01`–`run_04` 跳过已有的算例与
checkpoint，`run_07`、`run_09` 只补缺失部分，`run_05`、`run_06`、`run_08`、`run_10`
会清空自身产物后重算。已有部分结果时按 §0 单独运行剩下的脚本。

### 时间预算（4 核 CPU 实测，供排期用）

| 步骤 | 单位耗时 | 数量 | 合计 |
|---|---|---|---|
| `run_00` 冒烟 | — | 1 | ~5 min |
| `run_01` 生成算例 | — | 1 | < 1 s |
| `run_02` 主方法训练 | ~80 s/epoch × 250 epoch | 5 run | 长（见下） |
| `run_03` 消融训练 | 同上 | 11 变体 × 5 run | 长 |
| `run_04` 学习基线 | 同上 | 3 × 5 run | 长 |
| `run_05` 主评测 | ~9 s/算例·方法 | 21 方法 × 15 算例 | ~45 min |
| `run_07` CP-SAT 离线最优 | < 1 s/算例 | 24 | < 1 min |
| `run_07` HiGHS MILP | 2 s – 3600 s（时限）/算例 | 24 | 最坏 24 h 单进程，`--jobs 4` 约 6 h |
| `run_07` 在线重优化 | 1–80 s/算例 | 24 | ~10 min |
| `run_09` 训练 | 同 `run_02` | 9 run | 长 |
| `run_11`–`run_13` | — | — | 数分钟 |

训练是唯一的长项。**每个 run 用一个进程、`OMP_NUM_THREADS=1`，并行跑满核数**：网络很
小，torch 的多线程反而是负优化。例如 4 核上同时跑 3 个 run：

```
OMP_NUM_THREADS=1 python train.py --run-name fshgrl_run1 &
OMP_NUM_THREADS=1 python train.py --config ablation/nonoop.yaml --run-name nonoop_run1 &
OMP_NUM_THREADS=1 python train.py --config ablation/nobc.yaml --run-name nobc_run1 &
```

`checkpoint_best.pt` 每次验证（每 10 个 epoch）刷新一次，`checkpoint_last.pt` 每个
epoch 都刷新，因此**训练可以随时中断**，用当前的 best checkpoint 先评测出一版结果，
之后再续跑。

**若算力不足以按推荐预算跑完整个 run 矩阵，请采用等预算协议**：选一个所有方法都跑得
完的 epoch 数（例如 60），用 `--epochs` 传给每一个 run，主方法与全部消融、全部基线
一视同仁。等预算是可比性的前提——给主方法更多 epoch 再和基线比，是最容易被审稿人
一击命中的地方。届时论文里必须写明实际预算，不能说成推荐预算。想缩短单 run，改 `configs/algo.yaml` 的 `ppo.total_epochs`，
并把 `exploration.anneal_epochs` 同步改成它的约 35%（否则退火跑不完，
ε 会停在高位，策略被持续拉向均匀分布）。

若算力实在不够，`configs/env.yaml` 的 `fluid.resolve_every` 可以把流体 LP 的重解
节流到每 K 个决策点一次（LP 约占 rollout 用时的一半）。这会改变方法行为，
**不要用它跑主结果**，只作为稿件 §4.9 摊销成本的敏感性研究。

---

## 9. 在 PyCharm 中手动启动

所有 `scripts/run_*.py` 都是零参数脚本，PyCharm 里无需任何配置即可运行。

**方式 A（推荐）**：Project 视图里右键目标脚本（如 `scripts/run_02_train_main.py`）
→ **Run 'run_02_train_main'**。长训练建议用这种方式而不是终端——Run 窗口可随时停止、
可看完整输出、可加断点调试。

**方式 B（手动新建运行配置）**：Run → Edit Configurations → `+` → Python：

| 字段 | 取值 |
|---|---|
| Name | run_02_train_main |
| Script path | 工程根下的 `scripts/run_02_train_main.py` |
| Parameters | **留空**（脚本零参数） |
| Working directory | **工程根目录**（不是 `scripts/`） |
| Python interpreter | 项目虚拟环境解释器 |
| Environment variables | `PYTHONUNBUFFERED=1` |

需要串行执行多个配置时用 Run → Edit Configurations → `+` → **Compound**。
visdom 面板单独建一个配置：Script path 下拉切换为 **Module name**，填 `visdom.server`。

**常见坑**：

- **Working directory 填错**是最高频故障——填成 `scripts/` 会让相对路径全部
  FileNotFoundError。脚本内已 `os.chdir(工程根)` 兜底，但配置里仍应填对。
- PyCharm 内置终端在 Windows 上是 PowerShell，bash 的 `for ... done`、`$i`、
  行尾 `&` 在这里都会报错——本 README 的命令已全部避开这些语法，**照抄即可，不要改写**。
- 解释器别选成系统 Python：状态栏右下角确认是本项目 venv。
- 工程路径含空格不影响以上命令：全部使用相对路径，无需加引号。

---

## 10. 产物对照表（终检）

每个实验数据文件都有生成脚本，每个脚本的产物都有论文去向；反查无天窗。

| 实验数据文件 | 生成脚本（节号） | 服务论文哪张表/图/占位符 |
|---|---|---|
| `data/instances/index.csv` | `run_01`（§4） | 实例参数表的负荷指标行；占位符 `L0`–`L2` |
| `result/fshgrl_run*/log.csv` | `run_02`（§5） | 训练曲线图 (a)；占位符 `B1`、`R4` |
| `result/{noff,nofp,rp,hp,nofa,nosa,nohg,noall}_run*/log.csv` | `run_03`（§6） | 因子化消融表 T-NEW-3 |
| `result/{drlg,ahpdqn,hsddqn}_run*/log.csv` | `run_04`（§6） | 基线协议表 T-NEW-6；训练曲线图 (b) |
| `result/eval_results.csv` | `run_05`（§7） | 消融表、PDR 对比表、DRL 对比表；占位符 `A4`–`A6`、`R5` |
| `result/eval_ci.csv` | `run_11`（§8） | 上述三张表的 95% CI 列 |
| `result/pruning_stats.csv` | `run_06`（§7） | 剪枝量化表 T-NEW-4；占位符 `A1`、`A2`、`S2`–`S6`、`P-TIE` |
| `result/pruning_sensitivity.csv` | `run_06`（§7） | 图 F-NEW-2 |
| `result/exact_results.csv` | `run_07`（§0.2、§7） | 最优性间隙表 T-NEW-5：`eta_off_milp`/`eta_off_cpsat` 两列 → 表中 MILP / CP-SAT 列，`cpsat_time_s`、`milp_time_s` → 求解时间，`eta_online` → Online re-opt. 列；占位符 `A3`、`S7`、`S8`、`P-MILPCHK` |
| `result/arrival_results.csv` | `run_08`（§7） | 到达强度表 T-NEW-8(a)；占位符 `L3`、`L4` |
| `result/ood_results.csv` | `run_08`（§7） | OOD 表 T-NEW-8(b) |
| `result/shift_matrix.csv` | `run_08`（§7） | 图 F-NEW-4 |
| `result/reward_exploration.csv` | `run_09`（§0.3、§7） | 奖励/探索消融表 T-NEW-9（每行一个配置，`runs` 列注明是否复用主方法的 run）；占位符 `H4` |
| `result/case3d_results.csv` | `run_10`（§0.4、§7） | 案例研究表；占位符 `C-DDT600`、`C-DDT900`、`C-DDT1200` |
| `result/stats_summary.csv` | `run_11`（§8） | Wilcoxon/效应量表；占位符 `S1` |
| `result/variance_decomposition.csv` | `run_11`（§8） | Wilcoxon 表脚注的 ICC |
| `result/friedman_nemenyi.csv` | `run_11`（§8） | 图 F-NEW-5 |
| `result/figures/*.pdf` | `run_12`（§8） | 图 F-NEW-2 / F-NEW-4 / F-NEW-5 与两张面板图 |
| `result/paper_values.tex` | `run_13`（§8） | **全部 46 个 `\PH{}` 占位符** |

### 怎么把数据填回论文

1. 跑完 §4–§8，确认 `run_13` 打印 `[闭环] 论文全部占位符均已有数据来源。`
2. 把 `result/paper_values.tex` 复制到论文目录，在 `main.tex` 里 `\PH` 的定义之后加一行
   `\input{paper_values.tex}`。正文里的 `\PH{id}` 不用改：有数据的显示数值，缺数据的仍显示
   `[PH:id]`，所以 `run_13` 还没闭环时也可以先编译看效果。（旧说明让把 `\PH{A1}` 改成
   `\PHA1`，但 LaTeX 命令名只能由字母组成，`\PHA1` 会被读成 `\PHA` 后跟字符 1，现已改为
   按 id 查表。）
3. 表格里的蓝色 `\dc` 单元格按 §10 对照表找到对应 CSV，逐列填入。
   表头、单位与脚注都已写好，只需要填数值。

---

## 目录结构

```
configs/     参数中枢：instance / env / algo + ablation/*.yaml；代码无魔法数字
data/        算例生成（训练随机构造 + 评测逐个物化）与六档固定算例
environment/ 问题定义、交期感知流体松弛、离散事件环境（含剪枝与安全网）
agent/       异构图编码器、actor-critic、行为策略修正的 PPO、规则与学习基线
exact/       精确求解（CP-SAT 离线最优与在线重优化、HiGHS MILP）、MILP 证书与离散事件回放校验
analysis/    奖励恒等式校验、剪枝与 oracle 保留率、统计检验
result/      日志与全部实验 CSV、图、paper_values.tex
scripts/     零参数入口层：README 里的命令全部指向这里
docs/        实验设计规格（6A/6B/6C 契约）
```
