# 历史验收记录（不代表当前源码验收）

以下保留前次报告原文；当前结论以 [ACCEPTANCE.md](ACCEPTANCE.md) 为准。

# 代码定稿验收报告

验收日期：2026-09-25。范围仅为代码仓库；未访问或修改论文仓库，未启动正式训练，未优化已接受的方法或参数。

## 交付状态

- 固定基准已生成：50个唯一物理参数组合。验证6、主测试12、小规模12、长到达流12、大车间8；敏感性引用主测试文件。
- 正式清单已冻结：73个作业，64000000训练交互。当前正式账本已用交互为0，全部作业为not_started。
- 唯一配置为configs/experiment.yaml；固定SPT教师与已选择学习率直接写入，不依赖旧预实验或性能门槛。
- 代码按configs/data/environment/agent/result组织，保留十个真实入口。中文README、数据字典、实验规格、运行矩阵及主张映射齐备。
- 清理旧研究实现、日志、检查点、图表和报告11118个文件，约6.12 GiB；不建立历史归档。临时迁移材料及非最终工程运行目录在验收后清除，工程交互总账保留。

## 实测检查

1. 结构迁移时，11种策略的固定输入输出及一次关键参数更新与迁移前一致，最大数值误差为0；12步固定动作转移也一致。前后共24次环境交互已入账。
2. 最新回归测试31项全部通过，涵盖规则平局、DQN/DDQN目标、公开信息隔离、非抢占和部分加工后放弃、独立组件、节点置换/候选分块、情景共享及差分、KL回退、缓冲过期、六类训练恢复、缓存复用/损坏及完整参数网格。
3. 从空的独立工程结果目录完成17个微型训练配置，随后完成全部数据用途评测、共享状态串行计时、CP-SAT、统计与80个表图/数据资产导出。最终工程运行用时约115秒；这不是正式实验耗时估计。
4. 4096交互情景检查生成11个完整标签并发生非零偏好梯度，最大记录梯度范数约0.117。采用独立微型紧交期配置，仅检查监督链路；其数据与结果均不进入正式实验。
5. 最新微型流程复用了权重相同的检查点评测，实际完整评测110次、3831次环境交互，训练监测1148次交互，小规模离线回放12次。所有正式和微型评测记录都保留对应成本口径。
6. 从事件与状态增量重算履约率、等待动作/时间，核验工序时长、机器无重叠和累计占用；全部一致。已检查导出的甘特图和比较图预览，工程结果带明确水印。
7. Windows双进程启动与任务传递检查通过，未发生模拟交互。正式训练使用相同ProcessPool入口，未以正式矩阵开展并发训练测试。
8. 重复数据准备命令与重复冒烟命令验证复用，不新增模拟交互。所有源码/模型/实例身份检查保留；报告修改不触发重训。

本次全部工程自检累计44222次模拟交互，包括前期回归、修复前失败流程、最终微型流程和情景检查，低于50000上限。已修复的工程问题包括临时目录初始化、Windows长路径、缓存身份和计时统计；不存在未解决的验收失败。

## 正式运行前的资源信息

当次磁盘检查约有253.5 GiB可用。按微型轨迹、缓冲容量和检查点数量分项外推并增加两倍余量，预计约180.4 GiB，另要求5 GiB剩余保留空间。此为保守估算，正式体积与时间尚未实测；记录器持续检查可用容量，不自动删数据或降低粒度。

完整原始数据默认保存，包括训练实际算例、决策合法动作、状态增量、订单/工序事件、全部评测候选输出、情景实际未来和分支轨迹、教师快照、优化指标、计时样本、离线排程和来源清单。大型运行数据本地保留、Git忽略；固定50例随代码保留。

## 由用户启动

在README指定环境和项目根目录运行：

```powershell
python scripts/run_formal_all.py
```

此命令会真实启动6400万训练交互并按依赖顺序完成评测、统计和导出。也可按README逐节执行。完整正式性能、长期运行资源消耗及结果结论须由这次正式运行产生，当前验收不替代这些证据。

机器可读交付摘要见acceptance.json；本地详细验收及成本位于result/engineering，正式清单和零用量账本位于result/formal。

## 修订记录：2026-09-26 跨平台修订（不改验收结论）

本次修订不改变算法语义、参数与训练预算，不启动正式训练；改前改后用固定输入回归快照（`scripts/_regression_snapshot.py`，110 组：三个固定状态的观测数组、在线生成算例与微型基准算例数组、11 种方法正式规模网络的前向输出、11 种方法 32 步微型训练的最终权重与数值日志）逐位一致，回归测试 33 项全部通过（新增换行不变性与"恢复落在里程碑"两项）。

1. **根因与修复。** 固定算例与源码身份原按原始字节哈希，而 CSV 由默认 CRLF 行尾写出、git 存 LF，导致任何非 Windows-autocrlf 检出上 `run_00_smoke.py` 在数据校验步骤即失败。现在所有文本身份按换行归一后的内容计算（`result/storage.py:normalized_bytes`），CSV 显式以 LF 写出，新增 `.gitattributes` 强制文本文件 LF；`data/instances/fixed` 的 50 个算例内容逐字段不变，只刷新了 `sha256` 列、`manifest.json` 的文件哈希与配置指纹。
2. **身份值刷新。** `docs/acceptance.json` 的四个身份值由 `python scripts/_identity_report.py --write` 从当前提交重算：training_source `da4a8285…`、evaluation_source `2915bfc7…`、data_manifest_sha256 `d7520d69…`、smoke_identity `a89f316f…`；`tests_passed` 相应为 33。作者本机 2026-09-25 的账本用量（44222）与耗时记录保留不改。
3. **容器复核。** Linux、Python 3.11.15、torch 2.13.0、4 逻辑核、每进程 1 线程：完整冒烟 1 分 44 秒（`complete.json` 记录 101 秒），情景检查 11 个完整标签、非零偏好梯度；容器独立账本累计 28976 次交互（两次完整冒烟各约 1.3 万，其余为回归测试）。该账本与作者本机账本相互独立。
4. **统计协议（用户决定）。** 主证据改为种子向量自助区间与逐单元配对效应量（新增 `case_effects.csv`、`case_effects.pdf`，`components.tex` 增加正向种子数与胜/平/负列）；配对 Wilcoxon 原始 p 值降为附录参考（`tests.csv`、`tests.tex`），取消 Holm 校正。预算与作业矩阵不变。
5. **配置补全。** 训练与评测中的硬编码常量全部移入 `configs/experiment.yaml`（值不变）：`network.path_capacity/path_delivery`、`environment.exposure/exposure_threshold`、`data.due_jitter`、`demonstration.max_grad_norm`、`scenario.minimum_query_cost/reservation_floor/reservation_multiplier/wait_probe_interval`、`experiment.evaluate_milestones_from`、`recording.phase_startup_fraction`、`smoke.probe_due_factor`；`runtime.jobs/threads`、`smoke.interaction_limit/budget`、`recording.keep_recovery_copies` 现在真正生效，`runtime.device` 只允许 cpu。因新增键，`configs/experiment.py:identity(c)` 的值改变；当前不存在正式运行，无兼容性影响。冒烟数据目录若由旧配置指纹生成，会被改名保留后重新生成（`scripts/pipeline.py:retire_stale_smoke_data`）。
6. **清理与合并。** 删除未使用的属性、常量、参数与死存根（环境旧维度常量、`ActionConditionedGraph`、`Config.get/set/snapshot`、`full_source_hash`、`is_hopeless`、PPO 恒零的"scenario loss"及恒为 0/None 的日志字段 `scenario_loss/labelled_states/label_fraction/aux_gradient_norm/gradient_cosine/pair_rank_error`、`hold_id`、连续等待计数）；最短路径时间、交期构造与订单流元数据合并为 `data/generator.py` 的共享函数；PPO 与偏好更新的 KL 回退事务合并为 `agent/ppo.py:transactional_step`；网络状态哈希合并为 `result/storage.py:state_hash`；评测计划 `result/evaluation.py:evaluation_plan` 同时驱动评测与统计校验（不再产生 `sensitivity_changes.csv`、`sensitivity_default.csv` 中间表）；`agent/offline.py` 移至 `environment/offline_cpsat.py`；`result/evaluation.py` 纳入训练源码身份。
7. **命名。** 冻结的图策略统一称 `frozen`（原与 SPT 教师同用 `teacher`）：配置键 `scenario.frozen_interval`（原 `teacher_interval`），检查点键 `frozen_model/frozen_version/next_frozen`，日志列 `frozen_version`，记录流 `queries.frozen_policy`、对象目录 `artifacts/frozen_policies`，标签字段 `frozen_hash`。`teacher` 仅指规则教师。
8. **检查点保障。** 崩溃恢复中按上界计费的预留恰落在里程碑时补写该里程碑检查点；示范阶段结束恰在里程碑（1 万）时同样写出；预算在更新循环之前就已用尽时仍写出最终检查点。
9. **已知情况（未改）。** 观测构建、检查点复制、示范集重复序列化、经验缓冲整存、情景公开状态落盘、统计校验重哈希等 I/O 与计算开销见 README §9；配置里程碑 1 万仅作检查点，评测从 `experiment.evaluate_milestones_from`（10 万）起使用中间里程碑。

