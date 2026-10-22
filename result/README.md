# 结果目录说明

| 目录 | 内容 | 代码版本 |
|---|---|---|
| `formal/` | 第一轮正式实验（73 作业、6,400 万交互、50 例固定网格）的已发布结果：日志、评测、统计、表图；不含模型与原始轨迹。**原封保留**，分析见 [docs/results-analysis.md](../docs/results-analysis.md) | 标签 `paper2-v1-formal`（提交 `92f276a`）；用当前代码无法再通过其身份校验，这是预期的 |
| `pilot/` | 第二轮预注册试点（[docs/preregistration-v2.md](../docs/preregistration-v2.md)）：6 配置 × 3 种子 × 50 万交互，裁决在 `pilot_verdict.json` | 当前代码 |
| `formal2/` | 第二轮正式实验（试点通过后启动） | 当前代码 |
| `instance/` | 单算例收敛研究：12 个争用离线参照算例 × 6 配置 × 3 种子 × 10 万交互，曲线与最终目标值（`instance_curves.csv`、`instance_final.csv`、`paper_assets/instance_convergence.pdf`） | 当前代码 |
| `engineering/` | 工程冒烟与验收证据 | 各自目录记录身份 |

`.gitignore` 对 `pilot/`、`formal2/` 与 `instance/` 采用与 `formal/` 相同的规则：保留日志、评测与统计，忽略 `*.pt` 与 `raw/`。
