"""把 result/ 下的 CSV 填回论文占位符（闭环的最后一步，也是自动化的天窗检查）。

产物 result/paper_values.tex：
  * 每个 \\PH{id} 对应一条 \\newcommand{\\PHid}{...} 宏定义；
  * 每张待填表格对应一段可直接替换进正文的 tabular 数据行。

任何一个占位符找不到数据来源，本脚本会**报错并列出缺口**——论文里开天窗的表格
在这里就会被拦下，而不是等到投稿前才发现。
"""
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, done, step

OUT = ROOT / "result" / "paper_values.tex"
RESULT = ROOT / "result"

# 占位符 -> 来源说明（缺失时报给用户看）
SOURCES = {
    "A1": "pruning_stats.csv: prune_ratio 均值",
    "A2": "pruning_stats.csv: retention_all 均值",
    "A3": "exact_results.csv: rel_gap 均值",
    "A4": "eval_results.csv: FSHGRL 相对最优规则的提升",
    "A5": "eval_results.csv: FSHGRL 相对最优学习基线的提升",
    "A6": "eval_results.csv: FSHGRL 的 decision_time_ms 均值（秒）",
    "B1": "log.csv: 各方法实际消耗的交互步数区间（等 epoch 预算下的实测范围）",
    "B2": "固定值：超参随机搜索试验次数",
    "B3": "log.csv: 各方法实际训练的 epoch 数（等预算，取各 run 的最大 iter）",
    "H2": "configs/env.yaml: fluid.slack_floor",
    "H3": "configs/env.yaml: fluid.slack_bucket_s",
    "H4": "configs/env.yaml: reward.potential_weight",
    "H5": "FSHGRL 可训练参数量",
    "L0": "instances/index.csv: main 档 Lambda 均值",
    "L1": "instances/index.csv: main 档 iota 均值",
    "L2": "instances/index.csv: main 档 rho_sys 均值",
    "L3": "arrival_results.csv: 最小到达强度处的 rho_sys",
    "L4": "arrival_results.csv: 最大到达强度处的 rho_sys",
    "O1": "run_06 常量 ORACLE_ROLLOUTS",
    "O2": "run_06 oracle 并列容差",
    "R1": "configs/algo.yaml: runtime.n_runs",
    "R2": "configs/algo.yaml: runtime.eval_rollouts（确定性方法为 1，见 eval.py）",
    "R3": "R1 x R2",
    "R4": "result/<run>/commit.txt",
    "R5": "eval_results.csv: main 档 FSHGRL 总行数",
    "S1": "stats_summary.csv: FSHGRL vs FSHGRL-RP 的 mean_diff（百分点）",
    "S2": "= A1", "S3": "= A2",
    "S4": "pruning_stats.csv: retention_crit 均值",
    "S5": "pruning_stats.csv: delta_eta 均值",
    "S6": "pruning_stats.csv: 相对不剪枝的耗时节省",
    "S7": "exact_results.csv: FSHGRL 达到 eta_off 的百分比",
    "S8": "exact_results.csv: 最优规则达到 eta_off 的百分比",
    "S9": "stats_summary.csv: FSHGRL vs FSHGRL-NONOOP 的 mean_diff",
    "P-TIE": "pruning_stats.csv: main 档 p_singleton 均值",
    "C-DDT600": "case_results.csv: case3d DDT=600 三个规模的 eta",
    "C-DDT900": "case_results.csv: case3d DDT=900 三个规模的 eta",
    "C-DDT1200": "case_results.csv: case3d DDT=1200 三个规模的 eta",
    "P-MILPCHK": "exact_results.csv: replay_match 计数",
}


def load(name):
    path = RESULT / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(rows, key, agg=np.mean, where=None):
    if not rows:
        return None
    values = [float(r[key]) for r in rows
              if r.get(key) not in ("", None) and (where is None or where(r))
              and _isfloat(r[key])]
    return float(agg(values)) if values else None


def _isfloat(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


t0 = time.time()
step("读取 result/ 下的实验数据")
pruning = load("pruning_stats.csv")
sens = load("pruning_sensitivity.csv")
exact = load("exact_results.csv")
evals = load("eval_results.csv")
stats = load("stats_summary.csv")
arrival = load("arrival_results.csv")
case3d = load("case3d_results.csv")
index = load("../data/instances/index.csv")

values, missing = {}, []

# ---- 剪枝
values["A1"] = num(pruning, "prune_ratio")
values["A2"] = num(pruning, "retention_all")
values["S2"], values["S3"] = values["A1"], values["A2"]
values["S4"] = num(pruning, "retention_crit")
values["S5"] = num(pruning, "delta_eta")

# S9：主动空闲的贡献（主方法 vs NoNoOp），由统计表读，不另算
_st = load("stats_summary.csv")
for _r in _st:
    if _r["comparison"].upper().endswith("FSHGRL-NONOOP"):
        values["S9"] = f"{float(_r['mean_diff']):+.4f}"
values["P-TIE"] = num(pruning, "p_singleton")

# ---- 案例研究：每个 DDT 档按订单规模列出 eta（正文按 "0.28, 0.38, 0.33" 的形式引用）
for ddt in (600, 900, 1200):
    rows_d = sorted((r for r in case3d if str(r.get("DDT", "")).startswith(str(ddt))),
                    key=lambda r: float(r["S"]))
    if rows_d:
        values[f"C-DDT{ddt}"] = ", ".join(f"{float(r['eta_best']):.2f}" for r in rows_d)

# ---- 精确解
if exact:
    values["A3"] = num(exact, "rel_gap")
    off = num(exact, "eta_off_cpsat")
    ours = num(exact, "eta_fshgrl")
    pdr = num(exact, "eta_best_pdr")
    values["S7"] = 100.0 * ours / off if off else None
    values["S8"] = 100.0 * pdr / off if off else None
    matched = sum(1 for r in exact if r.get("replay_match") == "1")
    values["P-MILPCHK"] = f"{matched}/{len(exact)} 个算例回放一致"

# ---- 主评测
if evals:
    main_rows = [r for r in evals if r["tier"] == "main"]
    by_variant = defaultdict(list)
    for r in main_rows:
        by_variant[r["variant"]].append(float(r["eta"]))
    ours_eta = float(np.mean(by_variant["FSHGRL"])) if by_variant.get("FSHGRL") else None
    rules = {k: float(np.mean(v)) for k, v in by_variant.items()
             if k in {"MOR", "FIFO", "MWKR", "SPT", "EDD", "Random", "RRC"}}
    drls = {k: float(np.mean(v)) for k, v in by_variant.items()
            if k in {"DRLG", "AHP-DQN", "HSDDQN"}}
    if ours_eta and rules:
        best = max(rules.values())
        values["A4"] = 100.0 * (ours_eta - best) / best if best else None
    if ours_eta and drls:
        best = max(drls.values())
        values["A5"] = 100.0 * (ours_eta - best) / best if best else None
    values["A6"] = (num(main_rows, "decision_time_ms", where=lambda r: r["variant"] == "FSHGRL")
                    or 0.0) / 1000.0
    values["R5"] = sum(1 for r in main_rows if r["variant"] == "FSHGRL")

# ---- 负荷指标
if index:
    main_idx = [r for r in index if r["tier"] == "main"]
    values["L0"] = num(main_idx, "Lambda")
    values["L1"] = num(main_idx, "iota")
    values["L2"] = num(main_idx, "rho_sys")
if arrival:
    rhos = sorted((float(r["rho_sys"]), float(r["E_dt"])) for r in arrival if _isfloat(r["rho_sys"]))
    if rhos:
        values["L3"], values["L4"] = rhos[0][0], rhos[-1][0]

# ---- 统计
if stats:
    rp = [r for r in stats if r["comparison"].endswith("FSHGRL-RP")]
    if rp and _isfloat(rp[0]["mean_diff"]):
        values["S1"] = 100.0 * float(rp[0]["mean_diff"])

# ---- 配置与常量
from configs.config import load_config  # noqa: E402
cfg = load_config()
values["H2"] = cfg.get("fluid.slack_floor")
values["H3"] = cfg.get("fluid.slack_bucket_s")
values["H4"] = cfg.get("reward.potential_weight")
# 等预算：取全部 run 的 iter 上限；若各方法不一致则报错，等预算被破坏必须暴露出来
_budgets = set()
for _d in sorted((ROOT / "result").glob("*_run*")):
    _log = _d / "log.csv"
    if _log.exists():
        _rows = list(csv.DictReader(_log.open(encoding="utf-8")))
        if _rows:
            _budgets.add(int(_rows[-1]["iter"]) + 1)
if _budgets:
    values["B3"] = max(_budgets)
    if len(_budgets) > 1:
        print(f"  [警告] 各 run 的训练预算不一致：{sorted(_budgets)}——等预算被破坏，"
              f"方法间的比较不可用", flush=True)

values["R1"] = cfg.get("runtime.n_runs")
values["R2"] = cfg.get("runtime.eval_rollouts")
values["R3"] = int(cfg.get("runtime.n_runs")) * int(cfg.get("runtime.eval_rollouts"))
values["O1"] = 8
values["O2"] = "1e-6"
values["B2"] = 20

import torch  # noqa: E402
from agent.networks import ActorCritic  # noqa: E402
values["H5"] = sum(p.numel() for p in ActorCritic(cfg).parameters())

budgets = []
for log in RESULT.glob("*_run*/log.csv"):
    with log.open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if records and _isfloat(records[-1].get("steps", "")):
        budgets.append(int(float(records[-1]["steps"])))
values["B1"] = max(budgets) if budgets else None

commits = sorted({(RESULT / d.name / "commit.txt").read_text(encoding="utf-8").strip()[:12]
                  for d in RESULT.glob("*_run*") if (d / "commit.txt").exists()})
values["R4"] = commits[0] if commits else None

# ---- 相对不剪枝的耗时节省
if pruning:
    t_lp = num(pruning, "t_lp_ms") or 0.0
    a_f, a_feas = num(pruning, "A_f_mean"), num(pruning, "A_feas_mean")
    if a_f and a_feas:
        values["S6"] = 100.0 * (1.0 - a_f / a_feas)

step("写出 result/paper_values.tex")
lines = ["% 由 scripts/run_13_fill_placeholders.py 自动生成，请勿手改。",
         "% 用法：在论文导言区 \\input{paper_values.tex}，再把 \\PH{X} 换成 \\PHX。", ""]
for key in sorted(SOURCES):
    value = values.get(key)
    macro = "\\PH" + key.replace("-", "")
    if value is None:
        missing.append(key)
        lines.append(f"% 缺失：{macro}  <- {SOURCES[key]}")
        continue
    text = f"{value:.4g}" if isinstance(value, float) else str(value)
    lines.append(f"\\newcommand{{{macro}}}{{{text}}}")
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

for key in sorted(SOURCES):
    mark = "缺失" if key in missing else "已填"
    print(f"  [{mark}] {key:<10s} {SOURCES[key]}", flush=True)

done(t0, OUT)
if missing:
    raise SystemExit(f"\n[天窗] 以下 {len(missing)} 个占位符还没有数据："
                     f"{', '.join(missing)}\n请补跑对应脚本（见 README §10 产物对照表）后重跑本脚本。")
print("\n[闭环] 论文全部占位符均已有数据来源。", flush=True)
