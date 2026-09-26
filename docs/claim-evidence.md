# 主张—实现—实验—产物映射

正式实验尚未开始，以下是待验证主张，不预填优势结论。

| 待验证内容 | 实现 | 实验 | 原始记录 | 结果／表图 |
|---|---|---|---|---|
| 固定基准上的整体履约表现 | 完整方法 | 主比较、七规则、六学习对照 | main评测及订单结局 | per_case/per_seed/summary/case_effects（tests为附录）、comparison、fulfillment |
| 竞争修正的条件增益 | 局部竞争图、残差门控 | full与no_graph | 模型输出、机制快照 | effects/case_effects、components、mechanism |
| 情景策略改进的条件增益 | 可靠性加权偏好 | full与no_scenario | futures/branches/preferences/updates | effects/case_effects、preference_diagnostics |
| 示范与监督路径的作用 | 示范初始化、策略偏好头 | no_demo、impact_only、初始化教师比较 | 初始化检查点、示范轨迹 | initialization、components |
| 情景参数的影响 | count/weight/horizon | 六单因素变体与默认 | 相同主测试实例、参数及标签 | sensitivity |
| 不同规模的行为和成本 | 同一部署策略 | small/main/long_stream/large_shop | 每例轨迹、计时、实例参数 | generalization、runtime_parameters、latency |
| 与离线参照的距离 | CP-SAT时间映射 | 12小规模固定实例 | 排程、界、状态、回放 | offline_gaps、offline |
| 学习过程与资源代价 | 各训练器 | 固定里程碑和验证监测 | log、updates、预算、execution | learning_curves、costs |
| 派工/等待机制 | 完整动作空间 | 阶段与固定实例分析 | decisions/events/mechanisms | behavior、gantt、mechanism |

统计范围限于固定参数基准。每个物理参数组合只有一个实现，不能宣称估计了该参数的随机实例期望。五种子区间仅反映训练随机性；主证据是自助区间与逐单元效应量，配对Wilcoxon只作附录参考且不校正。两个核心消融识别条件增益；未增加纯示范HGNN匹配单元，不计算完整二乘二交互。失利数据与胜出数据同样进入所有完整性检查和导出。
