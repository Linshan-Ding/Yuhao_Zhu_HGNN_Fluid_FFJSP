"""所有 run_XX 脚本的公共前置：锚定工程根、统一计时/跳过/子进程调用、run 目录发现。"""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def training_budget() -> int:
    """训练预算的唯一真源：configs/algo.yaml 的 training.total_steps（环境交互步数）。"""
    from configs.config import load_config
    return int(load_config().get("training.total_steps"))


def step(title):
    print("\n" + "=" * 72 + f"\n[RUN] {title}\n" + "=" * 72, flush=True)


def done(t0, *outputs):
    print(f"[OK] 用时 {time.time() - t0:.1f}s", flush=True)
    for p in outputs:
        p = Path(p)
        print(f"     产物 {p.resolve()} {'[已落盘]' if p.exists() else '[缺失!]'}", flush=True)


def run_py(entry, *args):
    head = [entry] if entry.endswith(".py") else ["-m", entry]
    cmd = [sys.executable] + head + [str(a) for a in args]
    print("[CMD] " + " ".join(cmd), flush=True)
    if subprocess.call(cmd, cwd=str(ROOT)) != 0:
        sys.exit(f"[FAIL] {entry} 非零退出，后续步骤已中止")


# 冒烟/探针类 run 只跑几步，绝不能进评测与统计
EXCLUDED_RUN_PREFIXES = ("smoke", "probe", "debug", "tmp", "test")

# run 目录前缀 -> (训练方法, 论文里的名字, 配置叠加)。run 名 = 前缀 + _run{i}
RUN_SPECS = {
    "coh": ("coh", "CoH", ["coh.yaml"]),
    "coh_nohold": ("coh", "CoH-NoHold", ["ablation/nohold.yaml"]),
    "coh_nocritic": ("coh", "CoH-NoCritic", ["ablation/nocritic.yaml"]),
    "coh_nogate": ("coh", "CoH-NoGate", ["ablation/nogate.yaml"]),
    "coh_noholdcritic": ("coh", "CoH-NoHoldCritic", ["ablation/noholdcritic.yaml"]),
    "coh_nogatefeat": ("coh", "CoH-NoGateFeat", ["ablation/nogatefeat.yaml"]),
    "coh_edd": ("coh", "CoH-EDD", ["ablation/edd_exposure.yaml"]),
    "rulesel": ("rule_ppo", "RuleSel-PPO", ["baseline/rulesel_ppo.yaml"]),
    "tdqn": ("dqn", "Triplet-DQN", ["baseline/triplet_dqn.yaml"]),
    "hdqn": ("hdqn", "Hier-DQN", ["baseline/hier_dqn.yaml"]),
}
ABLATION_TAGS = [t for t in RUN_SPECS if t.startswith("coh_")]
BASELINE_TAGS = ["rulesel", "tdqn", "hdqn"]


def run_tag(name: str) -> str:
    return name.rsplit("_run", 1)[0]


def logged_steps(run_dir) -> int:
    path = Path(run_dir) / "log.csv"
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return int(rows[-1]["steps"]) if rows else 0


def run_complete(run_dir, budget=None) -> bool:
    budget = training_budget() if budget is None else int(budget)
    return logged_steps(run_dir) >= budget and (Path(run_dir) / "checkpoint_best.pt").exists()


def discover_runs(tags=None):
    """{run 前缀: [run 目录...]}，只收有 checkpoint_best.pt 且不属于冒烟前缀的目录。"""
    out = {}
    for d in sorted((ROOT / "result").glob("*_run*")):
        if not d.is_dir() or d.name.startswith(EXCLUDED_RUN_PREFIXES):
            continue
        tag = run_tag(d.name)
        if tag not in RUN_SPECS or (tags is not None and tag not in tags):
            continue
        if (d / "checkpoint_best.pt").exists():
            out.setdefault(tag, []).append(d)
    return out


def run_seed(index: int) -> int:
    """第 i 个独立 run 的种子：固定为 i，复现者不需要查表。"""
    return int(index)
