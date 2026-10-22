# 数据字典与恢复约定

## 实例与身份

固定基准为 data/instances/fixed2（旧 data/instances/fixed 仅供标签 `paper2-v1-formal` 复现，其 50 例在 fixed2 中 id 与文件字节完全相同）。cases中的CSV长表以kind区分header、stage_machines、proc、order、meta。订单和机器下标从0开始；时间单位为生成加工时间单位，不自动称为秒。CPU计时字段seconds为墙钟秒，ms为毫秒。

index.csv每固定单元一行：instance_id（可读名：前缀 V/M/S/L/H<阶段数>/CV/C/CS 加订单数，旧网格接 I<到达强度>-K<交期系数>，争用网格接 R<目标负荷>-K<交期系数>-r<复制序号>，如 M48-I1.5-K3.0、C96-R1.6-K2.0-r3）、legacy_id（第一轮的 case_<哈希> 名，用于对照 result/formal）、role、family、parameter_hash、file、sha256、orders/products/stages/machines_per_stage、processing、eligibility_probability、arrival_process、iota（争用族为由目标负荷反算的值）、due_factor、rho_target（争用族的目标负荷，旧族为 -1）、replicate（争用族的复制实例序号，旧族为 -1）、machines（机器总数）、catalog_sha256、seed、sampling_attempts、Lambda、empirical_Lambda、p_bar、rho_sys。manifest记录六族的随机子流、目标负荷与联合接受条件。sensitivity与contested引用同一ID列表。teacher.json记录在contested_validation上的教师择优（规则、各规则分数、数据指纹）。训练实例不混入固定测试索引。

所有sha256按换行归一后的内容计算（CRLF折为LF，二进制文件按原字节），因此同一提交在不同平台得到相同身份值；manifest的fingerprint是数据与基准配置段的哈希，配置键增减会改变它但不改变算例内容。

## 原始记录

每个训练作业及每次评测的raw目录下，manifest.json列出已提交chunks、instances和artifacts及SHA256。chunks/流名/*.jsonl.gz逐行可流式读取；result.recording.records负责校验并读取。表格CSV与压缩原始记录共同保留。

| 流 | 主键／关联 | 内容 |
|---|---|---|
| episodes | episode_id，instance对象哈希 | 初始状态、起始时间、已有事件、记录用途；恢复时可再次出现同ID的恢复快照 |
| decisions | episode_id + step | start/end、动作、全部合法legal_actions、等待订单/忙机器数、reward、done/terminated/truncated、phase、动态数组变更、主动空闲机器时间、environment_seconds；评测额外有全部候选评分与观测/推断时间 |
| events | episode_id + decision | arrival、operation_start/finish、order_resolved、wait；工序开始含order/stage/machine/duration/scheduled_end；wait 含被等机器 awaited_machine、代表订单 awaited_order 与预计完工 awaited_finish；结局含原因 |
| orders | episode_id + order + snapshot_time + episode_reason | 产品、到达、交期、outcome、status、stage、快照时间；预算收尾可能仍未解决，不强行记失败 |
| queries | query_id | 查询公开快照（public_state）、冻结策略对象（frozen_policy）、动作对（锚点=策略贪心动作，挑战者）、来源（contested_policy/contested_rule/explicit）、冻结策略版本 |
| futures | query_id + scenario | 实际模拟产品、到达、交期数组，不只是哈希 |
| branches | query_id + scenario + action_index | 分支episode_id，连接完整动作与事件轨迹 |
| preferences | query_id | 逐未来returns（按时完成数，[k_used,2]）、mean、standard_error、reliability、probability、complete、steps、seconds、k_used、winner（challenger/anchor/tie）、late_sets（锚点/挑战者承诺后预计迟交的订单）、冻结策略与候选策略哈希；两分支均由冻结策略延续 |
| updates | kind + epoch或number | 实际损失、梯度、KL、尝试/接受/回退及优化时间 |
| mechanisms | episode_id + phase | 阶段首个有效决策的公开图、动作、评分及表示对象引用 |

动态增量包含status、stage、machine_free_at、machine_busy_with、machine_busy_time、order_outcome的index/value数组；在初始快照上顺序应用即可重建状态。完整未来只在评测/训练记录器侧存储，策略观测仍只含公开状态。

outcome为1按时履约、0未履约、-1未解决。状态为0未到达、1等待、2加工、3完成、4放弃。开始加工不等于完成；正在加工的工序不可抢占。仅events中的实际operation_finish代表完成。放弃不回收已用产能。

DQN/DDQN的scores是Q值，probabilities和value为空；规则没有神经分数。未测量/不适用为JSON null或CSV空值，不解释为数值零。策略偏好概率与实际Wait动作、主动空闲时间分别记录。

## 场景与预算

收益差为挑战动作分支减锚点分支的按时完成订单数（整数，不再按队列规模归一；cohort_sizes 仅记录队列规模）。查询只在争用状态发出：候选承诺后“无法按时完成的订单集合”不同；共享未来逐个加入，直到 |mean|>2·standard_error、全部为零或达到 scenario.max_count。配对分支共享同一未来及冻结策略延续；reliability=|mean|/(|mean|+se+reliability_floor) 是训练启发式，不是统计置信保证。只有 winner=challenger 且 reliability≥scenario.reliability_threshold 的标签作为改进动作交叉熵目标进入更新；未完成分支的complete=false，保留成本与轨迹，不进入监督缓冲。训练日志的 usable_labels 累计可用标签数，contested_checks 累计争用判定次数。

训练日志每个 epoch 一行：total_steps=real_steps+scenario_steps+demonstration_steps+lost_upper_bound；eta_validation 为监控评测（争用验证集；单算例研究为该算例本身）的贪心履约率；episodes 为已完成的训练采样回合数，best_episode_eta 为其中到目前为止的最佳履约率，epoch_episode_eta 为本 epoch 内完成回合的平均履约率（无则为空）。记录器steps还包括该运行中的验证评测，不能直接当训练预算。预留账本在执行前持久化，恢复时未提交预留保守计费；未提交块和failed_raw清单保留但不混入已提交轨迹。

工程账本采用 engineering-accounting-2：`limit=null` 表示无累计硬上限，`historical_charged` 是原样保留的旧账本基数；`accounting/sessions/*.json` 每个进程独立写入 committed 与 charged_upper_bound，未正常结束的预留保守保留。汇总账本不会覆盖其他进程的消耗。该计数包含自检中的训练、监测、失败尝试和离线回放，不能当作正式训练预算。

完成作业的 completion_manifest.json 列出检查点、配置、源码包、日志、预算和 raw manifest 的哈希；复用前再核验原始块与对象。工程验收另记录 pytest.xml、进程 PID／执行区间、串并行模型比对和正式规模资源探针。资源探针中的合成奖励与偏好仅用于检查张量规模和更新路径，不进入正式统计。

训练checkpoint包含记录器提交点、环境、RNG、优化器、经验/标签缓冲、冻结策略与DQN目标网络；恢复检查点保留recording.keep_recovery_copies份（checkpoint_last、checkpoint_previous）。只有完整身份一致才可续训。校验失败不会覆盖已有数据。里程碑检查点checkpoint_里程碑.pt在training.milestones处写出（示范结束恰在1万时也写出）；崩溃恢复中按上界计费的预留恰落在里程碑时同样补写。

## 指标

eta为按时完成订单数/订单总数。wait_fraction为Wait动作占决策比例；held_share为主动空闲机器时间/总机器时间。startup为最后到达时刻前recording.phase_startup_fraction（0.2）的区间，arrivals为随后至最后到达，drain为收尾；阶段仅事后标注，时间区间跨界拆分。

decision_ms为平均网络/规则推断耗时，observation_ms为平均公开观测构建时间；decision_p50/p95_ms包含二者。environment_seconds仅环境事件推进。wall_seconds为整个调度评测，包括记录开销。recording_seconds累计序列化和写入计时；优化、构图等余项不能靠相减伪造。

latency.csv为SPT固定公共状态面板上的原始计时，不重复生成算例。orders/operations/machines/candidates描述该状态规模；seed是模型训练种子。规则seed=0仅是非训练标记，不增加重复。

统计以固定参数格上的训练种子向量为随机单位。旧网格相同参数一个实例，争用网格每单元 3 个独立实例（cell_effects 按设计单元给出复制实例均值的配对效应，分组 rho=/due=/orders= 按设计因子）；区间不覆盖随机算例生成波动。对照规则没有种子标准差。多方法的主比较以full为参照：summary/effects给出各条件组的自助95%区间与正向种子数、胜/平/负单元数，case_effects给出每个固定单元的配对种子均值差、种子标准差、自助区间与正向种子数（主证据）；tests只保留未校正的配对Wilcoxon原始p值作附录参考，有效样本数是训练种子数，不做Holm等多重比较校正。

## 单算例收敛研究

`result/instance/` 下：instance.csv 为各作业里程碑/最终检查点与七条规则在各自算例上的缓存评测；instance_curves.csv 为逐作业曲线（steps 已计费交互、epoch、real_steps、greedy_eta、best_episode_eta、epoch_episode_eta、episodes）；instance_final.csv 为每个作业的最终贪心履约率、最佳回合履约率、CP-SAT 可行解/上界与差距（gap_to_incumbent、gap_to_bound）；instance_summary.csv 按方法与指标给出 12 例均值的种子向量自助区间；instance_effects.csv 为 full 相对各对照的配对效应；instance_per_instance.csv 为逐算例的种子均值。exact 记录与试点根同身份时直接复用并标注 reused_from。

## 离线与来源

exact JSON保存每个工序的机器/开始/结束、求解器响应、时间映射、参数、上下界和回放，覆盖 experiment.exact_splits（small 与 contested_small）。只有OPTIMAL证明最优。可行解差为offline incumbent减online，可能为负，不误标为证明的最优性差距；gap_to_bound 为求解器上界减online，是任何策略都不可能超过的差距。

评测complete.json连接模型、固定实例和评测代码哈希。statistics_manifest连接输入/输出CSV；paper_assets/manifest连接每张表图。任何报告结论可沿这些引用回到原始订单和事件。

policy_sha256标识实际部署的网络结构与权重，checkpoint_sha256标识包含训练元数据的检查点文件。不同文件若部署权重一致，会复用同一算例的评测；文件来源仍分别保留。目录名采用短哈希以兼容Windows路径长度，记录内始终核对完整身份。失败评测attempt的交互也在evaluation_costs中单列。

trace_audit从原始状态增量和事件核验履约收益、等待比例、主动空闲时间、工序持续时间、机器无重叠及累计占用；甘特图直接使用相同工序事件。
