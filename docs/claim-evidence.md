# 主张—实现—实验—产物映射（v2）

v2 试点与正式实验尚未开始，以下是待验证主张，不预填优势结论。通过门槛见 [preregistration-v2.md](preregistration-v2.md)。

| 待验证内容 | 实现 | 实验 | 原始记录 | 结果／表图 |
|---|---|---|---|---|
| 争用基准上的整体履约表现 | 完整方法 | 争用测试主比较、七规则、六学习对照 | contested评测及订单结局 | per_case/per_seed/summary/case_effects/cell_effects（tests为附录）、comparison、fulfillment |
| 机器争用者注意力的条件增益 | 争用者注意力、争用计数门控 | full与no_graph | 模型输出、机制快照 | effects/case_effects/cell_effects、components、mechanism |
| 争用状态成对改进的条件增益 | 争用触发、策略锚点、自适应未来、交叉熵改进动作 | full与no_scenario | queries/futures/branches/preferences/updates | effects/case_effects、preference_diagnostics |
| 示范与监督路径的作用 | 验证集择优的教师示范、影响头 | no_demo、impact_only、初始化规则比较 | 初始化检查点、示范轨迹、teacher.json | initialization、components |
| 情景参数的影响 | max_count/reliability_threshold/auxiliary_weight | 六单因素变体与默认 | 相同争用测试实例、参数及标签 | sensitivity |
| 不同规模与旧网格上的行为和成本 | 同一部署策略 | contested/main/small/long_stream/large_shop | 每例轨迹、计时、实例参数 | generalization、comparison_legacy、runtime_parameters、latency |
| 与离线参照的距离 | CP-SAT时间映射与上界 | 24个小规模固定实例（旧12+争用12） | 排程、界、状态、回放 | offline_gaps（含gap_to_bound）、offline |
| 学习过程与资源代价 | 各训练器 | 固定里程碑和验证监测 | log、updates、预算、execution | learning_curves、fixed_budget_learning、costs |
| 等机派工与行为 | 完整动作空间（等机等待） | 阶段与固定实例分析 | decisions/events/mechanisms | behavior、gantt、mechanism |

统计范围限于固定参数基准。争用网格每个参数组合3个独立实例、旧网格1个实现，都不能宣称估计了该参数的随机实例期望。种子区间仅反映训练随机性；主证据是自助区间与逐单元效应量，配对Wilcoxon只作附录参考且不校正。两个核心消融识别条件增益；未增加纯示范HGNN匹配单元，不计算完整二乘二交互。失利数据与胜出数据同样进入所有完整性检查和导出。v1 的正式结果（result/formal）作为稿件附录，不与 v2 混合统计。
