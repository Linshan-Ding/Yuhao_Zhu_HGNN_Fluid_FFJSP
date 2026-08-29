"""主评测：FSHGRL + 全部消融变体 + 全部规则 + 全部学习基线，在 main 档逐算例评测。

自动发现 result/ 下的 checkpoint，复现者不需要填任何路径。
产物：result/eval_results.csv（逐算例一行）。
"""
import time

from _bootstrap import ROOT, checkpoints, done, run_py, step

from agent.baselines.rules import RULES

OUT = ROOT / "result" / "eval_results.csv"
TIERS = ["main"]
VARIANT_CONFIG = {
    "noff": "ablation/noff.yaml", "nofp": "ablation/nofp.yaml", "rp": "ablation/rp.yaml",
    "hp": "ablation/hp.yaml", "nofa": "ablation/nofa.yaml", "nosa": "ablation/nosa.yaml",
    "nohg": "ablation/nohg.yaml", "noall": "ablation/noall.yaml",
}
BASELINE_TAGS = {"drlg": "DRLG", "ahpdqn": "AHP-DQN", "hsddqn": "HSDDQN"}

t0 = time.time()
if OUT.exists():
    OUT.unlink()
    print(f"[INFO] 已清空旧的 {OUT.name}，评测结果将重新生成")

step("评测优先调度规则（无需 checkpoint）")
for rule in RULES:
    run_py("eval.py", "--method", rule, "--variant", rule, "--run-id", "rule",
           "--tiers", *TIERS, "--out", "result/eval_results.csv")

step("评测 FSHGRL 与全部消融变体")
found = checkpoints()
if not found:
    raise SystemExit("[FAIL] result/ 下没有 checkpoint，请先运行 run_02 / run_03 / run_04")
for ckpt in found:
    tag = ckpt.parent.name.rsplit("_run", 1)[0]
    run_id = ckpt.parent.name
    if tag in BASELINE_TAGS:
        continue
    extra = ["--config", VARIANT_CONFIG[tag]] if tag in VARIANT_CONFIG else []
    variant = "FSHGRL" if tag == "fshgrl" else f"FSHGRL-{tag.upper()}"
    run_py("eval.py", *extra, "--method", "FSHGRL", "--variant", variant,
           "--run-id", run_id, "--ckpt", str(ckpt), "--tiers", *TIERS,
           "--out", "result/eval_results.csv")

step("评测三个学习基线")
for ckpt in found:
    tag = ckpt.parent.name.rsplit("_run", 1)[0]
    if tag not in BASELINE_TAGS:
        continue
    method = BASELINE_TAGS[tag]
    run_py("eval.py", "--method", method, "--variant", method,
           "--run-id", ckpt.parent.name, "--ckpt", str(ckpt), "--tiers", *TIERS,
           "--out", "result/eval_results.csv")

done(t0, OUT)
