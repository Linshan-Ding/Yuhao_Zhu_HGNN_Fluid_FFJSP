# HGNN_Fluid_FFJSP

面向高频插单柔性流水车间调度的 HGNN-PPO 训练项目。当前版本面向 SCI 论文实验，训练侧不再依赖竞赛仿真接口，而是使用轻量数组仿真器采集完整 episode，以订单按时达成率为主优化目标。

## 核心目标

- 最大化订单按时达成率 `fulfillment_rate_mean`。
- 每个训练周期选择一个算例，并采集多条完整 episode。
- rollout 默认在 CPU 上推理，规避 Windows 显卡驱动 TDR 导致的系统重启。
- PPO 参数更新可使用 GPU，以保留训练吞吐。
- 训练日志实时写入 CSV，方便后续 SCI 实验画图和消融分析。

## 主要结构

```text
agent/
  train.py                 # SCI 实验训练入口
  sci_rollout.py           # 同步向量化完整 episode 采样
  training_simulator.py    # 轻量数组仿真器、算例解析与 cached_lp 流体特征
  PPO_model.py             # PPO 更新
  HGHH_model.py            # HGNN actor-critic，支持稀疏候选动作和可选候选 attention
  visualization.py         # Visdom 单指标窗口

data/
  config.json              # 默认训练配置
  config_annotation.json   # 配置说明
  instance/competition/    # 训练算例

result/
  train_metrics.csv        # 每个 iteration 实时追加一行
  last_checkpoint.pt       # 最近一次训练检查点
  best_checkpoint.pt       # 按 fulfillment_rate_mean 保存的最佳检查点
  ppo_policy_model.pt      # 最佳策略权重
```

## 运行

默认配置会从 `data/instance/competition` 读取 `DDT500_num50.txt`，可直接运行：

```bash
python agent/train.py
```

常用参数：

```bash
python agent/train.py --max-iterations 10
python agent/train.py --instances DDT500_num50.txt DDT600_num100.txt
python agent/train.py --instance-selection round_robin
python agent/train.py --rollout-device cpu --update-device cuda:0
python agent/train.py --resume none
```

`--device cuda:0` 仍兼容旧用法，等价于设置 PPO 更新设备。为了稳定，默认 `rollout_device` 保持为 `cpu`。

## 训练机制

- 每个 iteration 先按 `fixed`、`round_robin` 或 `random` 选择一个算例。
- 创建 `num_envs` 个数组仿真环境，每个环境跑完一个完整 episode。
- 同步收集所有可决策环境的观测，一次调用 `policy_old.act_batch()`。
- 每条轨迹只在最后一个 transition 标记 `done=True`，GAE 按 `trajectory_id` 分开计算。
- PPO 更新使用 `update_device`，默认可为 CUDA。
- `fluid_mode="cached_lp"` 时，仅在新订单到达或订单实际丢弃后重算 LP；Gurobi 不可用、超时或不可行时自动回退到 heuristic 特征。
- `use_sparse_attention=true` 时，只对当前合法 `(machine, task_type)` 候选做轻量 attention；`pair_order` scope 会再对选中 pair 下的 top-k 订单候选增强。

训练侧已弃用：

- `CompetitionPlatform.run_simulation()`
- `SchedulingAlgorithm.generate_schedule(platform)`
- `RolloutAlgorithm.generate_schedule()`
- `BatchedActor`

这些旧接口可以作为历史代码保留，但不再参与 SCI 训练主循环。

## 配置要点

`data/config.json` 中的关键字段：

- `simulator_backend`: 当前使用 `"array"`。
- `num_envs`: 每轮完整 episode 数；默认与 `num_workers` 一致。
- `rollout_device`: rollout 策略推理设备，默认 `"cpu"`。
- `update_device`: PPO 更新设备，默认 `"cuda:0"`，CUDA 不可用时自动回退 CPU。
- `resume`: `"auto"` 会从 `result/last_checkpoint.pt` 恢复。
- `checkpoint_interval`: 保存最近检查点的间隔。
- `train_instances`: 固定算例列表。
- `instance_selection`: `fixed`、`round_robin` 或 `random`。
- `order_top_k`: 每个工序类型保留的最紧急订单候选数。
- `fluid_mode`: `heuristic`、`cached_lp` 或 `off`；默认 `cached_lp`，无可用 Gurobi 时训练自动回退到 heuristic。
- `fluid_recompute_interval`: 兼容保留字段；事件驱动 `cached_lp` 不再按固定间隔重算。
- `fluid_time_limit`, `fluid_threads`: Gurobi 求解时间限制与线程数。
- `use_sparse_attention`: 是否启用稀疏合法候选 self-attention。
- `sparse_attention_scope`: `pair` 或 `pair_order`。
- `sparse_attention_heads`, `sparse_attention_dropout`: 稀疏候选 attention 的头数与 dropout。

`rollout_steps` 和 `target_transitions` 已废弃，完整 episode 是唯一采样单位。

## 日志指标

`result/train_metrics.csv` 每个训练周期实时追加一行。论文主曲线建议使用：

- `fulfillment_rate_mean`: 订单按时达成率主指标。
- `fulfillment_rate_std/min/max`: 同一周期多条 episode 的波动。
- `reward`, `mean_return`: PPO 代理训练信号。
- `loss`, `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction`: PPO 稳定性诊断。
- `rollout_seconds`, `update_seconds`, `policy_act_seconds`: 采样和更新耗时。
- `fluid_solve_count`, `fluid_solve_seconds`, `fluid_cache_hit_count`, `fluid_fallback_count`: 到达/丢弃触发后的 cached LP 求解、缓存与回退诊断。
- `sparse_attention_enabled`, `sparse_attention_scope`: 当前稀疏 attention 消融配置。
- `simulator_step_seconds`, `obs_build_seconds`: 数组仿真器内部耗时。
- `avg_action_tokens`, `max_action_tokens`, `padding_ratio`: 动作稀疏化效果。
- `cpu_memory_mb`, `gpu_memory_mb`, `gpu_temperature`: 资源监控。
- `completion_count`, `discard_count`: 完成和丢弃订单数。

## Visdom

启动 Visdom：

```bash
python -m visdom.server
```

训练时若 `viz=true`，每个数值指标使用独立窗口绘制，避免不同量纲曲线互相压扁。Visdom 不可用时训练不会中断，CSV 仍正常写入。

## 验证

```bash
python -m json.tool data/config.json
python -m json.tool data/config_annotation.json
python -m py_compile agent/train.py agent/sci_rollout.py agent/training_simulator.py agent/visualization.py agent/PPO_model.py agent/HGHH_model.py
```

短跑验证：

```bash
python agent/train.py --max-iterations 1 --rollout-device cpu --update-device cpu --resume none
```
