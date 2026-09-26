# 情景辅助竞争感知图强化学习：柔性流水车间调度

本仓库保留定稿方法及完整正式实验管线。当前方法与参数已经人工接受，不再自动调参。
**正式研究为 73 个训练作业、6400 万训练交互；本次重构不启动正式训练。**
每个参数组合只有一个固定算例。完整运行后，原始轨迹、统计结果和表图均保存在本仓库内，不依赖论文仓库或任何旧预实验文件。

## 1. 问题假设

订单动态到达、加工不可抢占、阶段顺序固定、机器具有加工资格。只有整个订单按时完成才计入等权履约收益；无望等待订单放弃，已经消耗的加工能力不返还。策略只能访问公开目录、已到达订单和可见历史，在线不运行规则教师或情景搜索。合法动作包括所有 Dispatch(order, machine) 与 Wait，候选不裁剪。

到达频率为 iota=lambda*p_bar，负荷为 rho=lambda*u_star。iota≥1 是本研究的操作性高频定义；同一目录下两者联动。主方法包含动作条件竞争图的门控残差修正、共享情景下可靠性加权策略改进；SPT 示范初始化是支持技术。

在线训练保持 3 产品、3 阶段、每阶段 2 台机器、48–96 订单、工时 4–30、资格概率 0.8、交期系数 1.5–3。固定评测采用每组合一个实例，训练仍在线随机生成。**固定算例、训练种子与同状态重复计时是三个不同概念。**

## 2. 环境配置

解释器固定为 `E:\anaconda3\envs\python3.13\python.exe`，项目根目录为 `D:\Python project\Yuhao_Zhu_HGNN_Fluid_FFJSP`。
在已经初始化 conda 的 PowerShell 中：

```powershell
conda activate python3.13
python -m pip install -r requirements.txt
python --version
```

依赖版本在 requirements.txt（下限版本；作者本机验证版本以注释标出）；已验证环境的完整元数据由运行目录 runtime.json 保存。下文 python 必须指向上述解释器。

**其他平台。** 代码本身与平台无关；在 Linux/macOS 上于仓库根目录执行 `python -m pip install -r requirements.txt` 后，把下文所有 PowerShell 命令原样在 shell 中运行即可（`python scripts/run_00_smoke.py` 等）。所有身份哈希（固定算例、源码、验收）按换行归一后的内容计算，`.gitattributes` 强制仓库内文本文件为 LF，因此同一提交在 Windows（autocrlf）与 Linux 上得到相同的身份值；`python scripts/_identity_report.py` 可打印当前检出的四个身份值并与 docs/acceptance.json 比对。容器复核环境：Linux、Python 3.11.15、torch 2.13.0、numpy 2.4.6、scipy 1.17.1、ortools 9.15.6755、matplotlib 3.11.1。

当前使用 CPU；并发作业数与每进程线程数由 configs/experiment.yaml 的 runtime.jobs（默认与上限，当前 8）和 runtime.threads（当前 1）决定，runtime.device 只允许 cpu。没有自动切换 GPU、异步算法或额外实时监控服务。唯一配置入口是 configs/experiment.yaml，加载、矩阵与身份规则见 configs/experiment.py；训练语义、情景查询、数据生成与记录阶段划分的全部常量都在该文件中，代码中不再有硬编码的算法常量。普通使用者不需要修改配置或脚本。

源码分为 configs、data、environment、agent、result 五模块。训练用源码签名（含 result/evaluation.py，它决定验证监测与最优检查点）、评测签名、报告来源分别保存；改报告不重训，改训练语义拒绝兼容续训。不要删除日志或修改冻结数据来绕过身份校验。

## 3. 冒烟自检

```powershell
python scripts/run_00_smoke.py
```

前置条件：依赖已安装。入口运行回归测试，再以独立微型配置遍历 11 个方法、6 个敏感性配置、六类数据用途、保存恢复、评测、CP-SAT、统计与图表导出，并执行有真实有效偏好更新的情景检查。

缩小网络与预算只用于工程验收，不能作为正式算法成绩。所有自检模拟调用（含测试、监测、失败和离线回放）纳入 result/engineering/interaction_ledger.json，累计上限为 smoke.interaction_limit（50000）；达到上限即停止，不能静默重置账本。源码修订后冒烟身份改变，需要重新完整冒烟（约 1.2 万次交互）；若本机账本已接近上限，请在确认修订内容后把 result/engineering/interaction_ledger.json 改名归档（保留文件），再运行冒烟，不要删除或改写账本数值。

成功标志：result/engineering/complete.json 与对应子目录 complete.json。已通过且源码身份一致时重复命令直接验证复用。失败查看控制台和该子目录 runs/作业名/stdout.log；修正工程错误后重复原命令，已匹配完成的产物可复用。若 data/instances/smoke 是由旧配置指纹生成的（例如配置键增减之后），冒烟会把它改名为 smoke_stale_指纹 保留并重新生成，不会删除任何数据。

耗时为本机实测自检耗时，最终记录在 complete.json。容器复核（Linux、4 逻辑核、每进程 1 线程）：回归测试 33 项全部通过，完整冒烟 1 分 49 秒，账本记入 13201 次交互（回归测试 1842、冒烟 11335，含情景检查）；作者本机实测见 docs/ACCEPTANCE.md。正式训练时间尚未实测，不能把微型耗时直接视为正式训练承诺。

## 4. 数据准备

```powershell
python scripts/run_01_prepare_data.py
```

配置来自 benchmark 与 data 部分，不训练模型。生成并校验 **50 个唯一参数组合**，保存在 data/instances/fixed/cases，每例一个 CSV。index.csv 与 manifest.json 记录完整参数、目录、实际频率/负荷、随机子流、文件哈希和用途映射。

| 用途 | 订单数／结构 | 算例数 |
|---|---|---:|
| 验证 | 标准车间，72 订单 | 6 |
| 主测试 | 标准车间，48、96 订单 | 12 |
| 小规模 | 标准车间，12、24 订单 | 12 |
| 长到达流 | 标准车间，128、192 订单 | 12 |
| 大车间 | 6 产品；5 阶段/128 订单及7阶段/192订单；每阶段3机器 | 8 |
| 敏感性 | 引用主测试文件，不新增算例 | 0 |

标准网格为 iota={0.6,1.5,2.5}、交期系数={1.5,3.0}；大车间只使用两档高频。标准测试族共享加工目录、订单流前缀、单位到达间隔和交期扰动。验证独立。目录须使整组频率下 rho 均在[0.3,1.8]，最多1000次尝试，失败不放宽条件。此为联合条件采样，不声称无条件均匀分布。

生成种子：验证1711001、标准测试1944001、大车间2166001/2166002。方法和训练种子不生成新测试算例。敏感性用途只是索引别名。

同时输出 result/formal/matrix.json、freeze.json、run_manifest.csv 和 budget_ledger.json。成功标志为50例及全部哈希通过校验、73作业合计6400万；重复命令仅复用。文件冲突报错，不能重新抽取有利样本。

## 5. 训练主方法

```powershell
python scripts/run_02_train_main.py
```

前置：数据准备完成，当前源码冒烟通过；训练入口会自动检查并在必要时运行冒烟。执行完整方法种子1–5，每个100万交互，共500万。结果位于 result/formal/runs/full_s1 至 full_s5。

固定教师SPT。每作业10000次示范交互，监督10轮、批256、学习率1e-3；随后PPO。图宽64、基础3层、局部2层、候选分块128；PPO学习率1e-4、rollout1024、minibatch128。默认情景数2、窗口2、辅助权重0.1、情景预算上限20%，其余有效参数均在配置文件。

示范、真实训练、情景及失败分支均计入该作业100万预算。按 training.milestones 保留 1 万（示范结束时刻）、10 万、30 万、50 万、100 万检查点，评测只使用 experiment.evaluate_milestones_from（10 万）起的中间里程碑；另保留初始化、验证最优与 recording.keep_recovery_copies 份恢复检查点（checkpoint_last、checkpoint_previous）。恢复包含优化器、随机状态、环境、经验与偏好缓冲、冻结策略快照、原始记录提交位置及预算；崩溃后被按上界计费的预留若恰好落在里程碑上，恢复时仍会写出该里程碑检查点。

成功标志：status.json 为 complete，最终检查点及原始记录通过核验。失败看 stdout.log、status.json 和 budget.json；重复原命令恢复。崩溃后无法确认的预留工作按上界计费并单列，不额外赠送训练预算。

## 6. 基线与消融

```powershell
python scripts/run_03_train_comparators.py
```

前置同主方法。执行六个学习对照与四个消融，各五种子、每作业100万，共50个作业、5000万交互。

学习对照为 DQN、DDQN、A2C、PPO、HGNN、双注意力。四经典算法采用相同公开特征、完整合法动作与普通MLP/集合聚合，DQN目标网络直接取最大值，DDQN在线选动作再由目标网络估值。已选学习率分别为3e-4、3e-4、3e-4、1e-4。不存在缩小正式基线网络或故意欠训练。

四消融为 no_graph、no_scenario、no_demo、impact_only，分别移除竞争结构、情景策略改进、示范，或改为独立影响头监督。所有结果保留；这些是条件组件比较，不是完整二乘二交互实验。

无需训练的规则在评测时执行：SPT、FCFS、EDD、MST、CR、LWKR，另保留 SPT-mixed（混合平局）附加对照。六条经典规则不主动等待；可派工时按显式规则选择。SPT-mixed保留原有候选排序下的最短工时平局行为（候选枚举顺序由 environment.exposure / exposure_threshold 决定，只影响这一变体的平局），具体定义见 agent/rules.py 与数据字典。

输出为 result/formal/runs/各方法_s种子；成功、失败和恢复规则同§5。

## 7. 评测与实验数据生成

```powershell
python scripts/run_04_train_sensitivity.py
python scripts/run_05_evaluate.py
python scripts/run_06_exact_reference.py
```

第一条：六个单因素非默认配置，种子1–3，各50万，共18作业、900万交互。情景数1/4、权重0.03/0.3、窗口1/4，默认2/0.1/2复用完整方法前三种子的50万检查点。与§5–6合计73作业、6400万。

第二条须全部训练完成：评测四类正式测试数据、核心中间检查点、验证最优检查点、初始化与教师，以及敏感性。相同模型/规则与同一算例的完整结果按哈希复用；不会因用途不同再次模拟。在线延迟使用SPT轨迹三个阶段首个有效公开状态面板，所有方法同状态、预热10次、串行计时30次。

输出 main.csv、small.csv、long_stream.csv、large_shop.csv、sensitivity.csv、main_里程碑.csv、main_best.csv、initialization.csv、latency.csv，以及 evaluations/哈希/ 下的完整原始记录。phase面板不存在的阶段不伪造状态。

第三条执行12个小规模CP-SAT：每例单线程、墙钟300秒、确定性时间60，求解时间上限合计至多约1小时（另有构模、回放和I/O）。只称OPTIMAL为已证明最优；其他状态保留可行解、上界或缺失。参考拥有未来信息，不能称为在线最优策略。输出 exact.csv、exact_schedule.csv、exact/*.json。

成功标志为评测与离线manifest及所有预定单元完整。失败/截断明确报错并保存已产生轨迹；重复命令复用身份匹配的完整结果，恢复缺失部分。

## 8. 统计聚合

```powershell
python scripts/run_07_statistics.py
python scripts/run_08_export.py
```

前置为§7全部完成。两条命令只消费已有记录，不隐式训练、调参或评测。

统计保留逐算例、逐种子结果；规则只有单一确定性结果。固定网格均值不解释为同参数随机实例期望。主证据是整体重采样训练种子向量得到的自助 95% 区间（summary/effects）与逐算例配对效应量（case_effects：种子均值差、种子标准差、自助区间、正向种子数）；配对 Wilcoxon 原始 p 值只作附录参考（tests.csv、tests.tex），不做 Holm 等多重比较校正——五个种子的最小双侧精确 p 值为 0.0625，显著性不是本研究的主张形式。不得把参数单元或计时重复当作独立训练重复。

第一条产出 summary/effects/case_effects/per_case/per_seed/tests/learning_curves/costs/behavior/offline_gaps 等CSV，并从原始事件核验履约率。第二条生成 result/formal/paper_assets 内的CSV、TeX、PDF、SVG、PNG与来源manifest；没有跨仓库写入。

导出检查数据、检查点、原始记录、预定单元及统计输入身份，不检查本文方法是否胜出。负结果保留。成功标志为paper_assets/manifest.json的全部资产与哈希一致。修改绘图后可仅重跑导出；修改统计需重新聚合，均不增加模拟交互。

## 9. 启动方式与并发度

### 9.1 PowerShell

在项目根目录完成环境配置后，可直接运行完整正式管线：

```powershell
python scripts/run_formal_all.py
```

**此命令会真实启动6400万训练交互**，依次完成自检、数据、训练、评测、离线参照、统计和导出。也可按§3–8逐阶段执行。

唯一可选参数是正整数并发上限，例如：

```powershell
python scripts/run_formal_all.py 4
```

并发上限为 runtime.jobs（8）个CPU作业，每进程 runtime.threads（1）线程；超出可用上限会提示夹取。数据、统计与导出串行，延迟单独串行。每个作业独立日志，父进程汇总。作业锁拒绝重复写入；正常完成自动释放，确认原进程已经退出的同主机残留锁可恢复。

正式总耗时尚未实测。execution目录保存每个批次的实际并行墙钟，costs.csv保存作业耗时；两者不能相互替代。storage_estimate.json分别按轨迹交互数、缓冲容量与检查点数量估算空间并增加余量；这不是已实测的正式体积。运行持续检查剩余容量，不足时中止并保留恢复检查点，不删数据或降低记录粒度。

**已知资源情况（2026-09-26 容器复核记录，未做优化）。** 每个决策的公开观测构建约 1.5–4.6 ms 且会重建一次 Problem 对象；每个 rollout 结束都完整保存恢复检查点并复制上一份；示范阶段每 256 步重新序列化增长中的示范集；DQN/DDQN 每 8 轮完整保存 5 万条经验；每次情景查询落一个公开状态 .pt；统计校验对每行评测重算 28 个源文件哈希。按微型冒烟外推，单作业 100 万交互约需 ≥3.3 小时，73 作业在 8 并发下 ≥26 小时；这是粗略外推而非实测承诺。

### 9.2 PyCharm

解释器选 `E:\anaconda3\envs\python3.13\python.exe`。Script path选择上述入口；Working directory选项目根目录；Parameters留空或单个正整数。可右键Run。无需填写config/path参数，不用临时循环，也不需要进入论文仓库。

### 9.3 保存与恢复

source.zip、config.yaml、runtime.json、manifest.json提供来源；checkpoint_last.pt和checkpoint_previous.pt为恢复状态。raw/manifest.json列出已提交压缩块和实例/策略对象，未提交或失败块另列，不能混入有效统计。

模型/环境/训练语义改变会拒绝原目录续训；并发度或报告修改不改变训练身份。不要并行复制或手工修改正在写入的运行目录。大型运行产物在Git中忽略，但本地完整保留。

## 10. 产物对照表

| 实验内容 | 生成入口 | 原始数据 → 统计／资产 |
|---|---|---|
| 固定参数基准 | run_01_prepare_data | cases/*.csv、index.csv、manifest.json |
| 主比较与参数影响 | run_05_evaluate、run_07_statistics、run_08_export | main.csv、逐案例轨迹 → per_case/summary/case_effects（tests 为附录）、comparison.tex、fulfillment/parameter_effects |
| 条件消融 | run_03_train_comparators、run_05_evaluate、run_07_statistics | 各消融评测 → effects/case_effects.csv、components.tex/pdf、case_effects.pdf |
| 敏感性 | run_04_train_sensitivity、run_05_evaluate | 同一主测试算例 → sensitivity.csv/tex/pdf |
| 规模与结构泛化 | run_05_evaluate | long_stream/large_shop.csv → generalization.tex/pdf |
| 离线参照 | run_06_exact_reference | 排程、界、求解状态 → offline_gaps.csv、offline.tex/pdf |
| 学习与初始化 | 训练入口、run_05_evaluate | log.jsonl、monitor、初始化及中间检查点 → learning_curves、initialization、learning.pdf、fixed_budget_learning.csv/pdf |
| 运行时间与决策时间 | 训练/评测入口 | updates、逐步计时、共享面板30次计时 → costs、latency、runtime_parameters |
| 等待与阶段行为 | run_05_evaluate | 决策/Wait事件及区间 → behavior.csv/tex/pdf |
| 竞争与情景机制 | 训练/评测入口 | 门控、评分、公开快照、真实模拟未来、分支及标签、冻结策略快照 → mechanism、preference_diagnostics |
| 甘特图与订单结局 | run_05_evaluate、run_08_export | operation_start/finish、orders → gantt.pdf/svg/png |
| 可追溯性与完整性 | run_07_statistics、run_08_export | 数据/模型/记录哈希 → trace_audit、statistics_manifest、paper_assets/manifest |

表中入口完整文件名均为 scripts/名称.py。原始字段、单位、缺失值和恢复规则见 [数据字典](docs/data-dictionary.md)；完整实验契约见 [实验规格](docs/experiment-spec.md)，主张映射见 [证据映射](docs/claim-evidence.md)。正式结果尚未生成，不预填胜出结论。
