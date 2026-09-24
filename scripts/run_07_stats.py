"""统计聚合：主判据、分层检验、非延迟的代价、异质性、方差分解、Friedman、预算核对、校准、精确参照汇总。

只读 result/ 下的 CSV，重算很快。判据按 docs/experiment-spec.md §7 预注册，这里只做机械核对。
产物（都在 result/）：stats_summary.csv、stratified_summary.csv、cell_means.csv、heterogeneity.csv、
variance_decomposition.csv、friedman_nemenyi.csv、budget_check.csv、ood_summary.csv、exact_summary.csv、
calibration.csv、gate_map.csv、verdict.csv。
"""
import csv
import time
from collections import defaultdict

import numpy as np

from _bootstrap import EXCLUDED_RUN_PREFIXES, ROOT, RUN_SPECS, done, run_tag, step, training_budget

from agent.baselines.rules import RULES
from analysis.stats import (band_permutation_test, bca_ci, cliffs_delta, exact_signflip_wilcoxon,
                            friedman_nemenyi, holm_bonferroni, intersection_union, variance_decomposition)
from result.logger import append_rows

RESULT = ROOT / "result"
OUTPUTS = ["stats_summary.csv", "stratified_summary.csv", "cell_means.csv", "heterogeneity.csv",
           "variance_decomposition.csv", "friedman_nemenyi.csv", "budget_check.csv", "ood_summary.csv",
           "exact_summary.csv", "calibration.csv", "gate_map.csv", "verdict.csv"]
MAIN, NOHOLD = "CoH", "CoH-NoHold"
DRL = ["RuleSel-PPO", "Triplet-DQN", "Hier-DQN"]
TIGHT, MID, LOOSE = 700.0, 1100.0, 1800.0


def read(name):
    path = RESULT / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(name, rows, columns):
    path = RESULT / name
    if path.exists():
        path.unlink()
    if rows:
        append_rows(path, rows, columns)


def paired_p(d):
    return exact_signflip_wilcoxon(d) if len(d) <= 20 else exact_signflip_wilcoxon(d)


def compare(name, ours, other, band=""):
    d = ours - other
    return {"comparison": name, "band": band, "n": int(d.size), "wins": int((d > 0).sum()),
            "ties": int((d == 0).sum()), "mean_diff": round(float(d.mean()), 5),
            "p_raw": paired_p(d), "cliff_delta": round(cliffs_delta(ours, other), 4),
            "ci_lo": "", "ci_hi": ""}


def main():
    t0 = time.time()
    for name in OUTPUTS:
        if (RESULT / name).exists():
            (RESULT / name).unlink()
    rows = read("eval_results.csv")
    if not rows:
        raise SystemExit("[FAIL] 缺 result/eval_results.csv，请先运行 run_05")

    # ------------------------------------------------------------ 逐算例均值（跨 run 与 rollout）
    grid = [r for r in rows if r["tier"] == "grid"]
    meta = {r["instance_id"]: (float(r["rho_target"]), float(r["DDT"]), int(r["S"])) for r in grid}
    instances = sorted(meta, key=lambda i: meta[i])
    per = defaultdict(lambda: defaultdict(list))               # variant -> instance -> [eta]
    held = defaultdict(lambda: defaultdict(list))
    for r in grid:
        per[r["variant"]][r["instance_id"]].append(float(r["eta"]))
        held[r["variant"]][r["instance_id"]].append(float(r["held_share"] or 0.0))
    rule_variants = [v for v in per if v in RULES or v.startswith("SPT-Idle(")]
    idle_variants = [v for v in rule_variants if v.startswith("SPT-Idle")]

    def means(variant):
        return np.asarray([float(np.mean(per[variant][i])) if per[variant].get(i) else np.nan for i in instances])

    table = {v: means(v) for v in per}
    # 派生对手：逐算例事后选最优阈值的 SPT-Idle*，逐算例最强规则 OracleRule，池化最强单条规则 BestRule
    if idle_variants:
        table["SPT-Idle*"] = np.nanmax(np.vstack([table[v] for v in idle_variants]), axis=0)
    if rule_variants:
        table["OracleRule"] = np.nanmax(np.vstack([table[v] for v in rule_variants]), axis=0)
        best_rule = max(rule_variants, key=lambda v: float(np.nanmean(table[v])))
        table["BestRule"] = table[best_rule]
    else:
        best_rule = ""
    if MAIN not in table:
        raise SystemExit("[FAIL] eval_results.csv 里没有 CoH 的行")
    ours = table[MAIN]
    drl_present = [v for v in DRL if v in table]
    best_drl = max(drl_present, key=lambda v: float(np.nanmean(table[v]))) if drl_present else ""

    # ------------------------------------------------------------ 池化比较族（Holm）
    step("池化配对比较（grid 档 36 算例，Holm 族 = 全部对比）")
    results = []
    for v in sorted(table):
        if v == MAIN:
            continue
        mask = np.isfinite(ours) & np.isfinite(table[v])
        if mask.sum() < 3:
            continue
        results.append(compare(f"{MAIN} vs {v}", ours[mask], table[v][mask]))
    holm = holm_bonferroni([r["p_raw"] for r in results])
    for r, h in zip(results, holm):
        r["p_holm"] = f"{h:.3e}"
        r["p_raw"] = f"{r['p_raw']:.3e}"
        d = ours - table[r["comparison"].split(" vs ", 1)[1]]
        d = d[np.isfinite(d)]
        lo, hi = bca_ci(d, n_boot=10000)
        r["ci_lo"], r["ci_hi"] = round(lo, 5), round(hi, 5)
        r["eta_ours"] = round(float(np.nanmean(ours)), 5)
        r["eta_other"] = round(float(np.nanmean(table[r["comparison"].split(' vs ', 1)[1]])), 5)
        print(f"  {r['comparison']:<32s} diff={r['mean_diff']:+.4f} wins={r['wins']}/{r['n']} "
              f"p_holm={h:.2e} delta={r['cliff_delta']:+.3f}", flush=True)
    write("stats_summary.csv", results, ["comparison", "band", "n", "wins", "ties", "mean_diff", "ci_lo",
                                         "ci_hi", "p_raw", "p_holm", "cliff_delta", "eta_ours", "eta_other"])
    lookup = {r["comparison"].split(" vs ", 1)[1]: r for r in results}

    # ------------------------------------------------------------ 分层比较
    step("分层比较（交期紧度三档、负荷两档；每族内 Holm）")
    bands = {"DDT700": np.asarray([meta[i][1] == TIGHT for i in instances]),
             "DDT1100": np.asarray([meta[i][1] == MID for i in instances]),
             "DDT1800": np.asarray([meta[i][1] == LOOSE for i in instances]),
             "rho<1": np.asarray([meta[i][0] < 1.0 for i in instances]),
             "rho>=1": np.asarray([meta[i][0] >= 1.0 for i in instances])}
    key_opponents = [v for v in ["SPT-Idle*", "OracleRule", "BestRule", NOHOLD, best_drl] if v and v in table]
    strat = []
    for band, mask in bands.items():
        fam = []
        for v in key_opponents:
            m = mask & np.isfinite(ours) & np.isfinite(table[v])
            fam.append(compare(f"{MAIN} vs {v}", ours[m], table[v][m], band))
        for r, h in zip(fam, holm_bonferroni([r["p_raw"] for r in fam])):
            r["p_holm"], r["p_raw"] = f"{h:.3e}", f"{r['p_raw']:.3e}"
            r["eta_ours"] = round(float(np.nanmean(ours[mask])), 5)
            r["eta_other"] = round(float(np.nanmean(table[r['comparison'].split(' vs ', 1)[1]][mask])), 5)
        strat.extend(fam)
        # 交–并检验：同时优于全部规则（含阈值网格）
        ps = []
        for v in rule_variants:
            m = mask & np.isfinite(ours) & np.isfinite(table[v])
            ps.append(paired_p((ours - table[v])[m]))
        strat.append({"comparison": f"{MAIN} beats all rules (IUT)", "band": band, "n": int(mask.sum()),
                      "wins": "", "ties": "", "mean_diff": "", "p_raw": f"{intersection_union(ps):.3e}",
                      "p_holm": "", "cliff_delta": "", "ci_lo": "", "ci_hi": "",
                      "eta_ours": round(float(np.nanmean(ours[mask])), 5), "eta_other": ""})
    ps_all = [paired_p(ours - table[v]) for v in rule_variants]
    strat.append({"comparison": f"{MAIN} beats all rules (IUT)", "band": "pooled", "n": len(instances),
                  "wins": "", "ties": "", "mean_diff": "", "p_raw": f"{intersection_union(ps_all):.3e}",
                  "p_holm": "", "cliff_delta": "", "ci_lo": "", "ci_hi": "",
                  "eta_ours": round(float(np.nanmean(ours)), 5), "eta_other": ""})
    write("stratified_summary.csv", strat, ["comparison", "band", "n", "wins", "ties", "mean_diff", "ci_lo",
                                             "ci_hi", "p_raw", "p_holm", "cliff_delta", "eta_ours", "eta_other"])
    for r in strat:
        if r["comparison"].endswith("(IUT)") or r["comparison"].endswith(NOHOLD):
            print(f"  [{r['band']:>7s}] {r['comparison']:<32s} diff={r['mean_diff'] or '':>8} p={r['p_raw']}", flush=True)

    # ------------------------------------------------------------ 每格均值与非延迟的代价
    step("每格（ρ × DDT）均值与非延迟的代价")
    cells = sorted({(meta[i][0], meta[i][1]) for i in instances})
    cell_rows = []
    for rho, ddt in cells:
        idx = [k for k, i in enumerate(instances) if meta[i][0] == rho and meta[i][1] == ddt]
        for v in sorted(table):
            vals = table[v][idx]
            h = [float(np.mean(held[v][instances[k]])) for k in idx if held[v].get(instances[k])]
            cell_rows.append({"rho": rho, "DDT": ddt, "variant": v, "n": len(idx),
                              "eta": round(float(np.nanmean(vals)), 5),
                              "held_share": round(float(np.mean(h)), 5) if h else "",
                              "price_vs_nohold": round(float(np.nanmean(ours[idx] - table[NOHOLD][idx])), 5)
                              if v == MAIN and NOHOLD in table else "",
                              "price_vs_oracle_rule": round(float(np.nanmean(ours[idx] - table["OracleRule"][idx])), 5)
                              if v == MAIN and "OracleRule" in table else ""})
    write("cell_means.csv", cell_rows, ["rho", "DDT", "variant", "n", "eta", "held_share",
                                        "price_vs_nohold", "price_vs_oracle_rule"])

    # ------------------------------------------------------------ 异质性
    het = []
    for opp in [v for v in [NOHOLD, "SPT-Idle*", "OracleRule"] if v in table]:
        d = ours - table[opp]
        for label, a, b in (("DDT700 vs DDT1800", bands["DDT700"], bands["DDT1800"]),
                            ("rho>=1 vs rho<1", bands["rho>=1"], bands["rho<1"])):
            sub = a | b
            obs, p = band_permutation_test(d[sub], a[sub])
            het.append({"comparison": f"{MAIN} vs {opp}", "contrast": label,
                        "diff_of_means": round(obs, 5), "p_perm": f"{p:.3e}"})
    write("heterogeneity.csv", het, ["comparison", "contrast", "diff_of_means", "p_perm"])

    # ------------------------------------------------------------ 方差分解、Friedman
    runs_by_inst = [[float(np.mean(per[MAIN][i]))] for i in instances]
    run_ids = sorted({r["run_id"] for r in grid if r["variant"] == MAIN})
    mat = np.asarray([[float(np.mean([float(r["eta"]) for r in grid if r["variant"] == MAIN
                                       and r["instance_id"] == i and r["run_id"] == rid]))
                       for rid in run_ids] for i in instances])
    vd = variance_decomposition(mat) if mat.shape[1] > 1 else {"var_instance": np.nan, "var_seed": np.nan,
                                                               "var_residual": np.nan, "icc_instance": np.nan,
                                                               "icc_seed": np.nan}
    write("variance_decomposition.csv", [
        {"source": "instance", "var_component": vd["var_instance"], "icc": vd["icc_instance"]},
        {"source": "seed", "var_component": vd["var_seed"], "icc": vd["icc_seed"]},
        {"source": "residual", "var_component": vd["var_residual"], "icc": ""}], ["source", "var_component", "icc"])
    fried_names = [v for v in [MAIN] + sorted(k for k in table if k.startswith("CoH-")) + drl_present
                   + ["SPT-Idle*", "OracleRule", "SPT"] if v in table]
    fmat = np.column_stack([table[v] for v in fried_names])
    fmask = np.isfinite(fmat).all(axis=1)
    fn = friedman_nemenyi(fmat[fmask], fried_names)
    write("friedman_nemenyi.csv", [{"method": m, "mean_rank": round(v, 4),
                                    "critical_difference": round(fn["critical_difference"], 4),
                                    "friedman_p": f"{fn['friedman_p']:.3e}", "n_instances": fn["n_instances"],
                                    "n_methods": fn["n_methods"]} for m, v in fn["mean_ranks"].items()],
          ["method", "mean_rank", "critical_difference", "friedman_p", "n_instances", "n_methods"])

    # ------------------------------------------------------------ 训练预算核对
    step("训练预算与稳定性核对")
    budget_rows = []
    for d in sorted(RESULT.glob("*_run*")):
        if not d.is_dir() or d.name.startswith(EXCLUDED_RUN_PREFIXES) or run_tag(d.name) not in RUN_SPECS:
            continue
        log = list(csv.DictReader((d / "log.csv").open(encoding="utf-8"))) if (d / "log.csv").exists() else []
        if not log:
            continue
        val = [(int(r["iter"]), int(r["steps"]), float(r["eta_val"])) for r in log if r.get("eta_val")]
        best = max(val, key=lambda x: x[2]) if val else (-1, 0, float("nan"))
        last10 = [v for _, _, v in val[-10:]]
        budget_rows.append({"run": d.name, "variant": RUN_SPECS[run_tag(d.name)][1], "steps": int(log[-1]["steps"]),
                            "epochs": len(log), "best_epoch": best[0], "best_steps": best[1],
                            "best_eta_val": round(best[2], 4),
                            "val_range_last10": round(max(last10) - min(last10), 4) if last10 else "",
                            "collapse_epochs": sum(int(r.get("collapse") or 0) for r in log),
                            "elapsed_s": float(log[-1]["elapsed_s"]),
                            "sps_collect": round(float(np.median([float(r["sps_collect"]) for r in log])), 1)})
    write("budget_check.csv", budget_rows, ["run", "variant", "steps", "epochs", "best_epoch", "best_steps",
                                            "best_eta_val", "val_range_last10", "collapse_epochs", "elapsed_s",
                                            "sps_collect"])
    budget = training_budget()
    short = [r["run"] for r in budget_rows if r["steps"] < budget]
    if short:
        print(f"  [警告] 未训满预算 {budget} 步的 run：{short}", flush=True)

    # ------------------------------------------------------------ OOD、精确参照、校准
    ood = [r for r in rows if r["tier"] == "ood"]
    ood_rows = []
    if ood:
        by = defaultdict(lambda: defaultdict(list))
        for r in ood:
            by[r["instance_id"]][r["variant"]].append(float(r["eta"]))
        for iid in sorted(by):
            v = by[iid]
            rule_best = max((k for k in v if k in RULES or k.startswith("SPT-Idle(")), key=lambda k: np.mean(v[k]), default="")
            drl_best = max((k for k in v if k in DRL), key=lambda k: np.mean(v[k]), default="")
            ood_rows.append({"instance_id": iid, "condition": iid.replace("ood_", ""),
                             "eta_coh": round(float(np.mean(v[MAIN])), 4) if MAIN in v else "",
                             "eta_nohold": round(float(np.mean(v[NOHOLD])), 4) if NOHOLD in v else "",
                             "eta_best_rule": round(float(np.mean(v[rule_best])), 4) if rule_best else "",
                             "best_rule": rule_best,
                             "eta_best_drl": round(float(np.mean(v[drl_best])), 4) if drl_best else "",
                             "best_drl": drl_best,
                             "held_share_coh": round(float(np.mean([float(r["held_share"]) for r in ood
                                                                    if r["instance_id"] == iid and r["variant"] == MAIN])), 4)
                             if MAIN in v else ""})
    write("ood_summary.csv", ood_rows, ["instance_id", "condition", "eta_coh", "eta_nohold", "eta_best_rule",
                                        "best_rule", "eta_best_drl", "best_drl", "held_share_coh"])

    exact = read("exact_results.csv")
    exact_rows = []
    if exact:
        f = lambda k: np.asarray([float(r[k]) for r in exact if r.get(k) not in ("", None)])
        coh, off, online = f("eta_coh"), f("eta_off"), f("eta_online")
        exact_rows.append({
            "n_instances": len(exact), "n_optimal": sum(r["cpsat_status"] == "OPTIMAL" for r in exact),
            "n_replay_match": sum(int(r["replay_match"]) for r in exact),
            "n_replay_match_online": sum(int(r["replay_match_online"]) for r in exact),
            "eta_off": round(float(off.mean()), 4), "eta_online": round(float(online.mean()), 4),
            "eta_coh": round(float(coh.mean()), 4) if coh.size else "",
            "eta_nohold": round(float(f("eta_nohold").mean()), 4) if f("eta_nohold").size else "",
            "eta_best_rule": round(float(f("eta_best_rule").mean()), 4),
            "eta_best_drl": round(float(f("eta_best_drl").mean()), 4) if f("eta_best_drl").size else "",
            "gap_coh_off": round(float((off - coh).mean()), 4) if coh.size else "",
            "gap_online_off": round(float((off - online).mean()), 4),
            "wins_coh_vs_online": int((coh > online[:coh.size]).sum()) if coh.size else "",
            "p_coh_vs_online": f"{exact_signflip_wilcoxon(coh - online[:coh.size]):.3e}" if coh.size else "",
            "online_time_s": round(float(f("online_time_s").mean()), 2),
            "cpsat_time_s": round(float(f("cpsat_time_s").mean()), 2)})
    write("exact_summary.csv", exact_rows, ["n_instances", "n_optimal", "n_replay_match", "n_replay_match_online",
                                            "eta_off", "eta_online", "eta_coh", "eta_nohold", "eta_best_rule",
                                            "eta_best_drl", "gap_coh_off", "gap_online_off", "wins_coh_vs_online",
                                            "p_coh_vs_online", "online_time_s", "cpsat_time_s"])

    trace = read("gate_trace.csv")
    cal_rows, map_rows = [], []
    if trace:
        disp = [r for r in trace if r["p_hat_chosen"] not in ("", None) and r["outcome"] not in ("", None)]
        p_hat = np.asarray([float(r["p_hat_chosen"]) for r in disp])
        y = np.asarray([float(r["outcome"]) for r in disp])
        edges = np.linspace(0, 1, 11)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p_hat >= lo) & (p_hat < hi) if hi < 1 else (p_hat >= lo) & (p_hat <= hi)
            if m.any():
                cal_rows.append({"bin_lo": round(lo, 1), "bin_hi": round(hi, 1), "n": int(m.sum()),
                                 "p_hat_mean": round(float(p_hat[m].mean()), 4),
                                 "outcome_rate": round(float(y[m].mean()), 4)})
                ece += m.mean() * abs(p_hat[m].mean() - y[m].mean())
        cal_rows.append({"bin_lo": "all", "bin_hi": "", "n": int(p_hat.size),
                         "p_hat_mean": round(float(((p_hat - y) ** 2).mean()), 4), "outcome_rate": round(float(ece), 4)})
        gated = [r for r in trace if r["p_hold"] not in ("", None)]
        pm = np.asarray([float(r["p_max"]) for r in gated])
        ea = np.asarray([float(r["exp_arrivals"]) for r in gated])
        ph = np.asarray([float(r["p_hold"]) for r in gated])
        hd = np.asarray([int(r["held"]) for r in gated])
        for i, (plo, phi) in enumerate(zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:])):
            for j, (elo, ehi) in enumerate(zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:])):
                m = (pm >= plo) & (pm < phi + (1e-9 if phi >= 1 else 0)) & (ea >= elo) & (ea < ehi + (1e-9 if ehi >= 1 else 0))
                if m.any():
                    map_rows.append({"p_max_lo": round(plo, 2), "p_max_hi": round(phi, 2), "exp_arr_lo": round(elo, 2),
                                     "exp_arr_hi": round(ehi, 2), "n": int(m.sum()),
                                     "p_hold_mean": round(float(ph[m].mean()), 4), "held_rate": round(float(hd[m].mean()), 4)})
    write("calibration.csv", cal_rows, ["bin_lo", "bin_hi", "n", "p_hat_mean", "outcome_rate"])
    write("gate_map.csv", map_rows, ["p_max_lo", "p_max_hi", "exp_arr_lo", "exp_arr_hi", "n", "p_hold_mean", "held_rate"])

    # ------------------------------------------------------------ 预注册判据
    step("预注册判据裁决（docs/experiment-spec.md §7）")
    verdict = []

    def judge(tag, opponent, need_delta=True):
        r = lookup.get(opponent)
        if r is None:
            verdict.append({"criterion": tag, "opponent": opponent, "verdict": "无数据", "detail": ""})
            return
        ok = float(r["p_holm"]) < 0.05 and r["mean_diff"] > 0 and (not need_delta or r["cliff_delta"] >= 0.33)
        verdict.append({"criterion": tag, "opponent": opponent, "verdict": "成立" if ok else "不成立",
                        "detail": f"diff={r['mean_diff']:+.4f} wins={r['wins']}/{r['n']} p_holm={r['p_holm']} delta={r['cliff_delta']:+.3f}"})
    judge("主判据：优于逐算例调优的 SPT-Idle*", "SPT-Idle*")
    judge("主判据：优于逐算例最强规则", "OracleRule")
    judge("非延迟的代价：优于同网络去等待", NOHOLD)
    if best_drl:
        judge("优于最强学习基线", best_drl)
    for tag in ("CoH-NoCritic", "CoH-NoGate", "CoH-NoHoldCritic", "CoH-NoGateFeat", "CoH-EDD"):
        judge(f"消融：{tag} 的贡献", tag, need_delta=False)
    for v in verdict:
        print(f"  [{v['verdict']:^4}] {v['criterion']:<28s} {v['opponent']:<16s} {v['detail']}", flush=True)
    write("verdict.csv", verdict, ["criterion", "opponent", "verdict", "detail"])
    done(t0, *[RESULT / n for n in OUTPUTS])


if __name__ == "__main__":
    main()
