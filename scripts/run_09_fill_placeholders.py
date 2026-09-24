"""把 result/ 里的统计结果写成论文可直接读入的 paper_values.tex 与 tables/*.tex。

    python scripts/run_09_fill_placeholders.py [--paper-dir ../Yuhao_Zhu_FFJSP_Order_GNN_Fluid_model]

SOURCES 的键集合与论文里 \\PH{} 的键集合必须逐一相等（论文仓库 README 给出核对命令）。
缺数据的键写成注释、表格照常生成，最后以非零码退出并列出缺口——论文里开天窗的数在这里被拦下。
"""
import argparse
import csv
import shutil
import subprocess
import time
from collections import defaultdict

import numpy as np

from _bootstrap import ROOT, done, step, training_budget

RESULT = ROOT / "result"
TABLES = RESULT / "tables"
VALUES = RESULT / "paper_values.tex"


def read(name):
    path = RESULT / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class Missing(Exception):
    pass


def need(rows, name):
    if not rows:
        raise Missing(name)
    return rows


# ------------------------------------------------------------------ 格式
def f3(x):
    return f"{float(x):.3f}"


def f2(x):
    return f"{float(x):.2f}"


def fp(p):
    p = float(p)
    return "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def fp_rel(p):
    """正文用：带关系符的 p 值（"<0.001" 或 "=0.03"），论文里写成 $p\\PH{...}$。表格单元用 fp()。"""
    text = fp(p)
    return text if text.startswith("<") else "=" + text


def sign(x):
    return f"{float(x):+.3f}"


# ------------------------------------------------------------------ 数据
class Data:
    def __init__(self):
        self.index = read("../data/instances/index.csv") if (ROOT / "data/instances/index.csv").exists() else []
        self.eval = read("eval_results.csv")
        self.stats = {r["comparison"].split(" vs ", 1)[1]: r for r in read("stats_summary.csv")}
        self.strat = defaultdict(dict)
        for r in read("stratified_summary.csv"):
            self.strat[r["band"]][r["comparison"]] = r
        self.cells = read("cell_means.csv")
        self.het = {(r["comparison"], r["contrast"]): r for r in read("heterogeneity.csv")}
        self.var = {r["source"]: r for r in read("variance_decomposition.csv")}
        self.fried = read("friedman_nemenyi.csv")
        self.budget = read("budget_check.csv")
        self.ood = read("ood_summary.csv")
        self.exact = read("exact_summary.csv")
        self.exact_rows = read("exact_results.csv")
        self.cal = read("calibration.csv")
        self.verdict = read("verdict.csv")

    def stat(self, opponent, key):
        r = self.stats.get(opponent)
        if r is None or r.get(key) in ("", None):
            raise Missing(f"stats_summary[{opponent}].{key}")
        return r[key]

    def strat_row(self, band, opponent, key):
        r = self.strat.get(band, {}).get(f"CoH vs {opponent}")
        if r is None or r.get(key) in ("", None):
            raise Missing(f"stratified[{band}][{opponent}].{key}")
        return r[key]

    def eta(self, variant, tier="grid"):
        vals = [float(r["eta"]) for r in self.eval if r["variant"] == variant and r["tier"] == tier]
        if not vals:
            raise Missing(f"eval[{variant}]")
        return float(np.mean(vals))

    def col(self, variant, key, tier="grid"):
        vals = [float(r[key]) for r in self.eval if r["variant"] == variant and r["tier"] == tier and r.get(key)]
        if not vals:
            raise Missing(f"eval[{variant}].{key}")
        return float(np.mean(vals))

    def cell(self, variant, key, rho=None, ddt=None):
        vals = [float(r[key]) for r in self.cells if r["variant"] == variant and r.get(key) not in ("", None)
                and (rho is None or float(r["rho"]) == rho) and (ddt is None or float(r["DDT"]) == ddt)]
        if not vals:
            raise Missing(f"cell[{variant}].{key}")
        return float(np.mean(vals))

    def held(self, band):
        rows = [r for r in self.eval if r["variant"] == "CoH" and r["tier"] == "grid"]
        if band.startswith("DDT"):
            rows = [r for r in rows if float(r["DDT"]) == float(band[3:])]
        elif band == "rho<1":
            rows = [r for r in rows if float(r["rho_target"]) < 1]
        else:
            rows = [r for r in rows if float(r["rho_target"]) >= 1]
        if not rows:
            raise Missing(f"held[{band}]")
        return float(np.mean([float(r["held_share"]) for r in rows]))

    def budget_coh(self, key, agg=np.median):
        vals = [float(r[key]) for r in self.budget if r["variant"] == "CoH" and r.get(key) not in ("", None)]
        if not vals:
            raise Missing(f"budget[CoH].{key}")
        return float(agg(vals))

    def exact_val(self, key):
        if not self.exact or self.exact[0].get(key) in ("", None):
            raise Missing(f"exact_summary.{key}")
        return self.exact[0][key]

    def ood_val(self, condition, key):
        for r in self.ood:
            if r["condition"] == condition:
                if r.get(key) in ("", None):
                    raise Missing(f"ood[{condition}].{key}")
                return r[key]
        raise Missing(f"ood[{condition}]")


def best_drl_name(d):
    names = [v for v in ("RuleSel-PPO", "Triplet-DQN", "Hier-DQN") if v in d.stats]
    if not names:
        raise Missing("best DRL")
    return max(names, key=lambda v: float(d.stats[v]["eta_other"]))


def best_rule_name(d):
    names = [v for v in d.stats if v not in ("SPT-Idle*", "OracleRule", "BestRule") and not v.startswith("CoH")
             and v not in ("RuleSel-PPO", "Triplet-DQN", "Hier-DQN")]
    if not names:
        raise Missing("best rule")
    return max(names, key=lambda v: float(d.stats[v]["eta_other"]))


def sources(d):
    n_grid = sum(r["tier"] == "grid" for r in d.index)
    bdrl = lambda: best_drl_name(d)
    brule = lambda: best_rule_name(d)
    src = {
        # 设置与协议
        "N-GRID": lambda: n_grid, "N-VAL": lambda: sum(r["tier"] == "val" for r in d.index),
        "N-SMALL": lambda: sum(r["tier"] == "small" for r in d.index),
        "N-OOD": lambda: sum(r["tier"] == "ood" for r in d.index),
        "R-RUNS": lambda: sum(r["variant"] == "CoH" for r in need(d.budget, "budget")),
        "B-STEPS-M": lambda: f"{training_budget() / 1e6:.1f}",
        "STEPS-RUN-M": lambda: f"{d.budget_coh('steps') / 1e6:.2f}",
        "T-RUN-MIN": lambda: f"{d.budget_coh('elapsed_s') / 60:.0f}",
        "SPS": lambda: f"{d.budget_coh('sps_collect'):.0f}",
        "BEST-STEPS-M": lambda: f"{d.budget_coh('best_steps') / 1e6:.2f}",
        "VAL-RANGE": lambda: f3(d.budget_coh("val_range_last10", np.mean)),
        "ICC-SEED": lambda: f2(need(d.var, "variance")["seed"]["icc"]),
        "ICC-INST": lambda: f2(d.var["instance"]["icc"]),
        "DT-COH": lambda: f2(d.col("CoH", "decision_time_ms")),
        "DT-SPT": lambda: f2(d.col("SPT", "decision_time_ms")),
        "N-FAMILY": lambda: len(need(d.stats, "stats")),
        "FRIEDMAN-P": lambda: fp_rel(need(d.fried, "friedman")[0]["friedman_p"]),
        "CD": lambda: f2(d.fried[0]["critical_difference"]),
        # 主结果（grid）
        "ETA-COH": lambda: f3(d.eta("CoH")), "ETA-NOHOLD": lambda: f3(d.eta("CoH-NoHold")),
        "ETA-SPT": lambda: f3(d.eta("SPT")), "ETA-SPTIDLE": lambda: f3(d.eta("SPT-Idle")),
        "ETA-SPTIDLESTAR": lambda: f3(d.stat("SPT-Idle*", "eta_other")),
        "ETA-ORACLERULE": lambda: f3(d.stat("OracleRule", "eta_other")),
        "NAME-BESTRULE": lambda: brule(), "ETA-BESTRULE": lambda: f3(d.stat(brule(), "eta_other")),
        "ETA-RULESEL": lambda: f3(d.eta("RuleSel-PPO")), "ETA-TDQN": lambda: f3(d.eta("Triplet-DQN")),
        "ETA-HDQN": lambda: f3(d.eta("Hier-DQN")),
        "NAME-BESTDRL": lambda: bdrl(), "ETA-BESTDRL": lambda: f3(d.stat(bdrl(), "eta_other")),
        "P-IUT": lambda: fp_rel(d.strat["pooled"]["CoH beats all rules (IUT)"]["p_raw"]),
        # 分层
        "P-HET-DDT": lambda: fp_rel(d.het[("CoH vs CoH-NoHold", "DDT700 vs DDT1800")]["p_perm"]),
        "P-HET-RHO": lambda: fp_rel(d.het[("CoH vs CoH-NoHold", "rho>=1 vs rho<1")]["p_perm"]),
        "HET-DDT": lambda: sign(d.het[("CoH vs CoH-NoHold", "DDT700 vs DDT1800")]["diff_of_means"]),
        "HELD-DDT700": lambda: f3(d.held("DDT700")), "HELD-DDT1800": lambda: f3(d.held("DDT1800")),
        "HELD-RHOLT1": lambda: f3(d.held("rho<1")), "HELD-RHOGE1": lambda: f3(d.held("rho>=1")),
        "NOOP-RATE": lambda: f3(d.col("CoH", "noop_rate")),
        # 校准
        "BRIER": lambda: f3([r for r in need(d.cal, "calibration") if r["bin_lo"] == "all"][0]["p_hat_mean"]),
        "ECE": lambda: f3([r for r in d.cal if r["bin_lo"] == "all"][0]["outcome_rate"]),
        # 精确参照
        "N-OPT": lambda: d.exact_val("n_optimal"), "N-REPLAY": lambda: d.exact_val("n_replay_match"),
        "ETA-OFF": lambda: f3(d.exact_val("eta_off")), "ETA-ONLINE": lambda: f3(d.exact_val("eta_online")),
        "ETA-COH-SMALL": lambda: f3(d.exact_val("eta_coh")), "ETA-NOHOLD-SMALL": lambda: f3(d.exact_val("eta_nohold")),
        "ETA-BESTRULE-SMALL": lambda: f3(d.exact_val("eta_best_rule")),
        "GAP-COH-OFF": lambda: f3(d.exact_val("gap_coh_off")), "GAP-ONLINE-OFF": lambda: f3(d.exact_val("gap_online_off")),
        "W-COH-ONLINE": lambda: f"{d.exact_val('wins_coh_vs_online')}/{d.exact_val('n_instances')}",
        "P-COH-ONLINE": lambda: fp_rel(d.exact_val("p_coh_vs_online")),
        "T-ONLINE": lambda: f"{float(d.exact_val('online_time_s')):.0f}",
        "T-CPSAT": lambda: f"{float(d.exact_val('cpsat_time_s')):.1f}",
        # OOD
        "ETA-COH-OOD": lambda: f3(np.mean([float(r["eta_coh"]) for r in need(d.ood, "ood")])),
        "ETA-BESTRULE-OOD": lambda: f3(np.mean([float(r["eta_best_rule"]) for r in d.ood])),
        "W-COH-OOD": lambda: f"{sum(float(r['eta_coh']) > float(r['eta_best_rule']) for r in d.ood)}/{len(d.ood)}",
        "ETA-COH-S500": lambda: f3(d.ood_val("S500", "eta_coh")),
        "ETA-BESTRULE-S500": lambda: f3(d.ood_val("S500", "eta_best_rule")),
        "ETA-COH-MMPP": lambda: f3(d.ood_val("mmpp", "eta_coh")),
        "ETA-BESTRULE-MMPP": lambda: f3(d.ood_val("mmpp", "eta_best_rule")),
    }
    # 池化对比：增益、胜场、Holm p、Cliff δ
    for key, opp in (("SPTIDLESTAR", "SPT-Idle*"), ("ORACLERULE", "OracleRule"), ("NOHOLD", "CoH-NoHold")):
        src[f"GAIN-{key}"] = (lambda o=opp: sign(d.stat(o, "mean_diff")))
        src[f"W-{key}"] = (lambda o=opp: f"{d.stat(o, 'wins')}/{d.stat(o, 'n')}")
        src[f"P-{key}"] = (lambda o=opp: fp_rel(d.stat(o, "p_holm")))
        src[f"D-{key}"] = (lambda o=opp: sign(d.stat(o, "cliff_delta")))
    src["GAIN-BESTDRL"] = lambda: sign(d.stat(bdrl(), "mean_diff"))
    src["W-BESTDRL"] = lambda: f"{d.stat(bdrl(), 'wins')}/{d.stat(bdrl(), 'n')}"
    src["P-BESTDRL"] = lambda: fp_rel(d.stat(bdrl(), "p_holm"))
    src["D-BESTDRL"] = lambda: sign(d.stat(bdrl(), "cliff_delta"))
    # 分层：非延迟的代价与对最强规则的增益
    for band in ("DDT700", "DDT1100", "DDT1800", "RHOLT1", "RHOGE1"):
        b = {"RHOLT1": "rho<1", "RHOGE1": "rho>=1"}.get(band, band)
        src[f"PRICE-{band}"] = (lambda bb=b: sign(d.strat_row(bb, "CoH-NoHold", "mean_diff")))
        src[f"GAINR-{band}"] = (lambda bb=b: sign(d.strat_row(bb, "OracleRule", "mean_diff")))
        src[f"W-NOHOLD-{band}"] = (lambda bb=b: f"{d.strat_row(bb, 'CoH-NoHold', 'wins')}/{d.strat_row(bb, 'CoH-NoHold', 'n')}")
        src[f"W-ORACLE-{band}"] = (lambda bb=b: f"{d.strat_row(bb, 'OracleRule', 'wins')}/{d.strat_row(bb, 'OracleRule', 'n')}")
        src[f"P-NOHOLD-{band}"] = (lambda bb=b: fp(d.strat_row(bb, "CoH-NoHold", "p_holm")))
        src[f"P-ORACLE-{band}"] = (lambda bb=b: fp(d.strat_row(bb, "OracleRule", "p_holm")))
    # 每格极值
    def cell_extreme(which):
        rows = [r for r in d.cells if r["variant"] == "CoH" and r.get("price_vs_nohold")]
        if not rows:
            raise Missing("cell price")
        r = (max if which == "max" else min)(rows, key=lambda x: float(x["price_vs_nohold"]))
        return r
    src["PRICE-MAX"] = lambda: sign(cell_extreme("max")["price_vs_nohold"])
    src["CELL-MAX"] = lambda: f"$\\rho={float(cell_extreme('max')['rho']):.1f}$, DDT {int(float(cell_extreme('max')['DDT']))}"
    src["PRICE-MIN"] = lambda: sign(cell_extreme("min")["price_vs_nohold"])
    src["CELL-MIN"] = lambda: f"$\\rho={float(cell_extreme('min')['rho']):.1f}$, DDT {int(float(cell_extreme('min')['DDT']))}"
    # 消融
    for key, var in (("NOCRITIC", "CoH-NoCritic"), ("NOGATE", "CoH-NoGate"), ("NOHOLDCRITIC", "CoH-NoHoldCritic"),
                     ("NOGATEFEAT", "CoH-NoGateFeat"), ("EDD", "CoH-EDD")):
        src[f"A-{key}"] = (lambda v=var: sign(d.stat(v, "mean_diff")))
        src[f"P-{key}"] = (lambda v=var: fp_rel(d.stat(v, "p_holm")))
    return src


# ------------------------------------------------------------------ 表格
def tabular(colspec, header, body, path, note=None):
    lines = [f"% generated by scripts/run_09_fill_placeholders.py ({time.strftime('%Y-%m-%d')}); do not edit",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    lines += [" & ".join(h) + " \\\\" for h in header]
    lines.append("\\midrule")
    lines += [" & ".join(str(c) for c in row) + " \\\\" for row in body]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"% {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def band_eta(d, variant, ddt):
    vals = [float(r["eta"]) for r in d.eval if r["variant"] == variant and r["tier"] == "grid" and float(r["DDT"]) == ddt]
    return f3(np.mean(vals)) if vals else "--"


def method_row(d, variant, label=None):
    label = label or variant
    if variant == "CoH":
        return [label, f3(d.eta("CoH"))] + [band_eta(d, "CoH", x) for x in (700.0, 1100.0, 1800.0)] + ["--", "--", "--", "--"]
    r = d.stats.get(variant)
    if r is None:
        return None
    etas = [band_eta(d, variant, x) for x in (700.0, 1100.0, 1800.0)] if variant in {x["variant"] for x in d.eval} \
        else ["--"] * 3
    return [label, f3(r["eta_other"])] + etas + [sign(r["mean_diff"]), f"{r['wins']}/{r['n']}", fp(r["p_holm"]), sign(r["cliff_delta"])]


def write_tables(d):
    TABLES.mkdir(parents=True, exist_ok=True)
    header = [["Method", "$\\eta$", "DDT 700", "DDT 1100", "DDT 1800", "CoH $-$ method", "Wins", "Holm $p$", "Cliff's $\\delta$"]]
    # 规则表
    rules = ["MOR", "FIFO", "MWKR", "SPT", "EDD", "Random", "RRC", "SPT-Idle", "SPT-Idle(1.0)", "SPT-Idle(1.25)",
             "SPT-Idle(2.0)", "SPT-Idle(3.0)", "SPT-Idle*", "OracleRule"]
    body = [row for row in (method_row(d, v, {"SPT-Idle*": "SPT-Idle$^{*}$ (per-instance $\\theta$)",
                                              "OracleRule": "Best rule per instance"}.get(v)) for v in rules) if row]
    body.insert(0, method_row(d, "CoH"))
    tabular("l" + "c" * 8, header, body, TABLES / "tab_rules.tex", "grid tier, 36 instances, instance means over runs")
    # 学习方法表
    body = [method_row(d, "CoH")] + [row for row in (method_row(d, v) for v in ("CoH-NoHold", "RuleSel-PPO", "Triplet-DQN", "Hier-DQN")) if row]
    tabular("l" + "c" * 8, header, body, TABLES / "tab_drl.tex")
    # 消融表
    body = [method_row(d, "CoH")] + [row for row in (method_row(d, v) for v in
            ("CoH-NoHold", "CoH-NoCritic", "CoH-NoGate", "CoH-NoHoldCritic", "CoH-NoGateFeat", "CoH-EDD")) if row]
    tabular("l" + "c" * 8, header, body, TABLES / "tab_ablation.tex")
    # 代价表：每格 CoH / NoHold / 最强规则
    rhos = sorted({float(r["rho"]) for r in d.cells})
    ddts = sorted({float(r["DDT"]) for r in d.cells})
    head = [["$\\rho_{\\mathrm{sys}}$"] + [f"\\multicolumn{{3}}{{c}}{{DDT {int(x)}}}" for x in ddts],
            [""] + ["CoH", "Non-delay", "Best rule"] * len(ddts)]
    body = []
    for rho in rhos:
        row = [f"{rho:.1f}"]
        for ddt in ddts:
            for v in ("CoH", "CoH-NoHold", "OracleRule"):
                try:
                    row.append(f3(d.cell(v, "eta", rho, ddt)))
                except Missing:
                    row.append("--")
        body.append(row)
    tabular("l" + "c" * (3 * len(ddts)), head, body, TABLES / "tab_price.tex")
    # 分层表
    body = []
    for band in ("DDT700", "DDT1100", "DDT1800", "rho<1", "rho>=1"):
        for opp, label in (("CoH-NoHold", "learned non-delay"), ("SPT-Idle*", "SPT-Idle$^{*}$"), ("OracleRule", "best rule")):
            r = d.strat.get(band, {}).get(f"CoH vs {opp}")
            if r:
                body.append([band.replace("rho", "$\\rho$").replace(">=", "$\\geq$").replace("<", "$<$"), label, r["n"],
                             sign(r["mean_diff"]), f"{r['wins']}/{r['n']}", fp(r["p_holm"]), sign(r["cliff_delta"])])
    tabular("llccccc", [["Band", "CoH vs", "$n$", "Mean diff.", "Wins", "Holm $p$", "Cliff's $\\delta$"]], body,
            TABLES / "tab_stratified.tex")
    # 精确参照表（按 S 汇总）
    body = []
    if d.exact_rows:
        for S in sorted({int(r["S"]) for r in d.exact_rows}):
            rows = [r for r in d.exact_rows if int(r["S"]) == S]
            g = lambda k: f3(np.mean([float(r[k]) for r in rows if r.get(k)])) if any(r.get(k) for r in rows) else "--"
            body.append([S, len(rows), sum(r["cpsat_status"] == "OPTIMAL" for r in rows), g("eta_off"), g("eta_online"),
                         g("eta_coh"), g("eta_nohold"), g("eta_best_rule"), g("eta_best_drl")])
    tabular("l" + "c" * 8, [["$S$", "Inst.", "Optimal", "Offline opt.", "Rolling re-opt.", "CoH", "Non-delay", "Best rule", "Best DRL"]],
            body, TABLES / "tab_exact.tex")
    # OOD 表
    body = [[r["condition"].replace("_", "\\_"), r["eta_coh"], r["eta_nohold"], f"{r['eta_best_rule']} ({r['best_rule']})",
             f"{r['eta_best_drl']} ({r['best_drl']})" if r["best_drl"] else "--", r["held_share_coh"]] for r in d.ood]
    tabular("lccccc", [["Condition", "CoH", "Non-delay", "Best rule", "Best DRL", "Held share"]], body, TABLES / "tab_ood.tex")
    # 算例表
    body = []
    for tier in ("grid", "val", "small", "ood"):
        rows = [r for r in d.index if r["tier"] == tier]
        if rows:
            body.append([tier, len(rows), "/".join(sorted({r["S"] for r in rows}, key=int)),
                         "/".join(sorted({str(int(float(r["DDT"]))) for r in rows}, key=int)),
                         f"{min(float(r['rho_sys']) for r in rows):.2f}--{max(float(r['rho_sys']) for r in rows):.2f}"])
    tabular("lcccc", [["Tier", "Instances", "$S$", "DDT", "Realised $\\rho_{\\mathrm{sys}}$"]], body, TABLES / "tab_instances.tex")
    # 预算表
    by = defaultdict(list)
    for r in d.budget:
        by[r["variant"]].append(r)
    body = []
    for v, rows in sorted(by.items()):
        med = lambda k: float(np.median([float(x[k]) for x in rows if x.get(k) not in ("", None)]))
        body.append([v, len(rows), f"{med('steps') / 1e6:.2f}", f"{med('best_steps') / 1e6:.2f}",
                     f3(np.mean([float(x["val_range_last10"]) for x in rows if x.get("val_range_last10")])) if any(x.get("val_range_last10") for x in rows) else "--",
                     f"{med('elapsed_s') / 60:.0f}", f"{med('sps_collect'):.0f}"])
    tabular("lcccccc", [["Method", "Runs", "Steps (M)", "Best ckpt (M)", "Val. range", "Minutes", "Steps/s"]], body, TABLES / "tab_budget.tex")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", default=None)
    args = parser.parse_args()
    t0 = time.time()
    d = Data()
    step("生成 tables/*.tex")
    write_tables(d)
    step("生成 paper_values.tex")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    lines = [f"% generated by scripts/run_09_fill_placeholders.py at commit {commit}; do not edit",
             "\\makeatletter", "\\newcommand{\\PHset}[2]{\\@namedef{PHval@#1}{#2}}",
             "\\DeclareRobustCommand{\\PH}[1]{\\@ifundefined{PHval@#1}{{\\color{revblue}\\textbf{[PH:#1]}}}{\\@nameuse{PHval@#1}}}",
             "\\makeatother"]
    missing = []
    src = sources(d)
    for key in sorted(src):
        try:
            value = src[key]()
            lines.append(f"\\PHset{{{key}}}{{{value}}}")
        except (Missing, KeyError, IndexError, ValueError, ZeroDivisionError) as exc:
            missing.append(f"{key} ({exc})")
            lines.append(f"% missing: {key}")
    VALUES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {len(src) - len(missing)}/{len(src)} 个占位符已填", flush=True)
    if args.paper_dir:
        paper = ROOT / args.paper_dir if not str(args.paper_dir).startswith("/") else args.paper_dir
        paper = type(ROOT)(paper)
        shutil.copy(VALUES, paper / "paper_values.tex")
        (paper / "tables").mkdir(exist_ok=True)
        for f in TABLES.glob("*.tex"):
            shutil.copy(f, paper / "tables" / f.name)
        figures = RESULT / "figures"
        if figures.exists():
            (paper / "figures").mkdir(exist_ok=True)
            for f in figures.glob("*.pdf"):
                shutil.copy(f, paper / "figures" / f.name)
        print(f"  已复制 paper_values.tex、tables/*.tex 与 figures/*.pdf 到 {paper}", flush=True)
    done(t0, VALUES, TABLES)
    if missing:
        raise SystemExit(f"\n[天窗] 以下 {len(missing)} 个占位符还没有数据：\n  " + "\n  ".join(missing))
    print("[闭环] 论文全部占位符均已有数据来源。", flush=True)


if __name__ == "__main__":
    main()
