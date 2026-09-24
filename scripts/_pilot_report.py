"""CoH 试点裁决：按 docs/experiment-spec.md §9 预注册的规则机械核对，不做事后调整。

1. 每个 coh_p*_run* 的 checkpoint_best 在 main 档贪心评测一次 -> result/pilot_eval.csv（可续跑）；
2. 八条规则在同一环境设置（configs/coh/p0_backbone.yaml：可救优先暴露、无流体）下评测
   -> result/pilot_rules.csv；
3. 从每个 run 的 log.csv 取最佳验证 epoch、末 10 次验证极差、总交互步数；
4. 按规则打印裁决表，写 result/pilot_summary.csv。

用法：python scripts/_pilot_report.py [--refresh]
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

from _bootstrap import ROOT, run_py, step

from agent.baselines.rules import RULES
from result.logger import append_rows
from run_14_pilot_coh import PILOT

EVAL = ROOT / "result" / "pilot_eval.csv"
RULES_OUT = ROOT / "result" / "pilot_rules.csv"
SUMMARY = ROOT / "result" / "pilot_summary.csv"
TIGHT_MAX, LOOSE_MIN = 1100.0, 1400.0
SUMMARY_COLUMNS = ["config", "runs", "eta_pooled", "eta_tight", "eta_loose", "wins_vs_best_rule",
                   "val_range_last10", "best_epoch", "steps", "held_share_tight", "held_share_loose",
                   "decision_time_ms"]


def read(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_dirs():
    out = defaultdict(list)
    for d in sorted((ROOT / "result").glob("coh_p*_run*")):
        if (d / "checkpoint_best.pt").exists():
            out[d.name.split("_")[1]].append(d)
    return out


def evaluate_missing(runs, refresh):
    have = {(r["variant"], r["run_id"]) for r in read(EVAL)} if not refresh else set()
    if refresh and EVAL.exists():
        EVAL.unlink()
    for tag, dirs in runs.items():
        for d in dirs:
            if (f"CoH-{tag.upper()}", f"{d.name}_r1") in have:
                continue
            run_py("eval.py", "--config", PILOT[tag], "--method", "FSHGRL",
                   "--variant", f"CoH-{tag.upper()}", "--run-id", d.name,
                   "--ckpt", str(d / "checkpoint_best.pt"), "--tiers", "main", "--rollouts", "1",
                   "--out", "result/pilot_eval.csv")
    if refresh or not RULES_OUT.exists():
        if RULES_OUT.exists():
            RULES_OUT.unlink()
        for rule in RULES:
            run_py("eval.py", "--config", PILOT["p0"], "--method", rule, "--variant", rule,
                   "--run-id", "rule", "--tiers", "main", "--out", "result/pilot_rules.csv")


def log_diagnostics(d):
    rows = read(d / "log.csv")
    val = [(int(r["iter"]), float(r["eta_val"])) for r in rows if r.get("eta_val")]
    best = max(val, key=lambda x: x[1]) if val else (-1, float("nan"))
    last = [v for _, v in val[-10:]]
    return {"best_epoch": best[0], "val_range": (max(last) - min(last)) if last else float("nan"),
            "steps": int(rows[-1]["steps"]) if rows else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="重评全部 run 与规则")
    args = parser.parse_args()
    runs = run_dirs()
    if not runs:
        raise SystemExit("[FAIL] 没有 result/coh_p*_run*/checkpoint_best.pt，请先运行 run_14_pilot_coh.py")

    step("评测试点 checkpoint 与规则（已评过的跳过）")
    evaluate_missing(runs, args.refresh)

    step("汇总（docs/experiment-spec.md §9）")
    rows = read(EVAL)
    ddt = {r["instance_id"]: float(r["DDT"]) for r in rows}
    per = defaultdict(lambda: defaultdict(list))       # tag -> instance -> [eta per run]
    held = defaultdict(lambda: defaultdict(list))
    dt = defaultdict(list)
    for r in rows:
        tag = r["variant"].split("-")[1].lower()
        per[tag][r["instance_id"]].append(float(r["eta"]))
        held[tag][r["instance_id"]].append(float(r.get("held_share") or 0.0))
        dt[tag].append(float(r["decision_time_ms"]))
    rule_rows = read(RULES_OUT)
    for r in rule_rows:
        ddt.setdefault(r["instance_id"], float(r["DDT"]))
    rule_eta = defaultdict(lambda: defaultdict(list))
    for r in rule_rows:
        rule_eta[r["instance_id"]][r["variant"]].append(float(r["eta"]))
    instances = sorted(ddt, key=lambda i: (ddt[i], i))
    best_rule = {i: max((np.mean(v) for v in rule_eta[i].values()), default=float("nan"))
                 for i in instances}
    tight = [i for i in instances if ddt[i] <= TIGHT_MAX]
    loose = [i for i in instances if ddt[i] >= LOOSE_MIN]

    metrics = {}
    for tag in sorted(per):
        m = {i: float(np.mean(per[tag][i])) for i in instances if per[tag].get(i)}
        diag = [log_diagnostics(d) for d in runs.get(tag, [])]
        metrics[tag] = {
            "runs": len(runs.get(tag, [])),
            "pooled": float(np.mean([m[i] for i in instances if i in m])),
            "T": float(np.mean([m[i] for i in tight if i in m])),
            "L": float(np.mean([m[i] for i in loose if i in m])),
            "wins": int(sum(m[i] > best_rule[i] for i in instances if i in m)),
            "V": float(np.mean([d["val_range"] for d in diag])) if diag else float("nan"),
            "best_epochs": [d["best_epoch"] for d in diag],
            "steps": int(np.mean([d["steps"] for d in diag])) if diag else 0,
            "held_T": float(np.mean([np.mean(held[tag][i]) for i in tight if held[tag].get(i)])),
            "held_L": float(np.mean([np.mean(held[tag][i]) for i in loose if held[tag].get(i)])),
            "dt": float(np.mean(dt[tag])), "per_instance": m,
        }
        print(f"  {tag}: runs={metrics[tag]['runs']} pooled={metrics[tag]['pooled']:.4f} "
              f"tight={metrics[tag]['T']:.4f} loose={metrics[tag]['L']:.4f} "
              f"wins vs best rule {metrics[tag]['wins']}/{len(instances)} "
              f"val_range={metrics[tag]['V']:.3f} best_epochs={metrics[tag]['best_epochs']} "
              f"held tight/loose={metrics[tag]['held_T']:.3f}/{metrics[tag]['held_L']:.3f} "
              f"{metrics[tag]['dt']:.2f} ms", flush=True)
    if rule_rows:
        print(f"  最强规则（同一环境设置，逐算例取最大）池化 {np.mean([best_rule[i] for i in instances]):.4f}",
              flush=True)

    def adopt(cand, ref):
        """预注册规则：紧档均值 +0.02 且 >=7/9 胜，或验证极差减半且紧档不低于对照 -0.01；
        宽档比对照低 >0.03 一票否决。"""
        if cand not in metrics or ref not in metrics:
            return None, "缺数据"
        c, r = metrics[cand], metrics[ref]
        wins = sum(c["per_instance"][i] > r["per_instance"][i] for i in tight
                   if i in c["per_instance"] and i in r["per_instance"])
        gain = c["T"] - r["T"]
        cond_a = gain >= 0.02 and wins >= 7
        cond_b = np.isfinite(c["V"]) and np.isfinite(r["V"]) and c["V"] <= 0.5 * r["V"] \
            and c["T"] >= r["T"] - 0.01
        guard = c["L"] >= r["L"] - 0.03
        detail = (f"紧档 {gain:+.4f}（{wins}/{len(tight)} 胜）；验证极差 {c['V']:.3f} vs {r['V']:.3f}；"
                  f"宽档 {c['L'] - r['L']:+.4f}")
        if not guard:
            return False, detail + "；宽档代价 >0.03，否决"
        return (cond_a or cond_b), detail

    print()
    winner = "p0"
    for cand, ref in (("p1", "p0"), ("p2", "p1")):
        ok, detail = adopt(cand, ref)
        if ok is None:
            print(f"  [缺数据] {cand.upper()} vs {ref.upper()}", flush=True)
            continue
        print(f"  [{'采纳' if ok else '不采纳'}] {cand.upper()} vs {ref.upper()}：{detail}", flush=True)
        if ok and winner == ref:
            winner = cand
    print(f"  第一波胜者：{winner.upper()}", flush=True)
    if "p3" in metrics:
        c, r = metrics["p3"], metrics[winner]
        ok = c["T"] >= r["T"] - 0.01 and c["L"] >= r["L"] - 0.03
        print(f"  [{'采纳' if ok else '不采纳'}] P3（纯 on-policy）vs {winner.upper()}：紧档 "
              f"{c['T'] - r['T']:+.4f}，宽档 {c['L'] - r['L']:+.4f}（平局取简）", flush=True)
    if "p0" in metrics and metrics["p0"]["pooled"] < 0.70:
        print("  [提示] P0 池化低于 0.70（NoAll 水平）：按 §9 加跑 P0 + 动作自注意力 3 run 作骨干候选",
              flush=True)

    if SUMMARY.exists():
        SUMMARY.unlink()
    append_rows(SUMMARY, [{
        "config": tag, "runs": m["runs"], "eta_pooled": round(m["pooled"], 4),
        "eta_tight": round(m["T"], 4), "eta_loose": round(m["L"], 4),
        "wins_vs_best_rule": m["wins"], "val_range_last10": round(m["V"], 4),
        "best_epoch": " ".join(str(e) for e in m["best_epochs"]), "steps": m["steps"],
        "held_share_tight": round(m["held_T"], 4), "held_share_loose": round(m["held_L"], 4),
        "decision_time_ms": round(m["dt"], 3)} for tag, m in sorted(metrics.items())],
        SUMMARY_COLUMNS)
    print(f"\n  产物 {EVAL}\n  产物 {RULES_OUT}\n  产物 {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
