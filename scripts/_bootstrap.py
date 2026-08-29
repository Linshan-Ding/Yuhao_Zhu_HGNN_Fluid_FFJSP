"""所有 run_XX 脚本的公共前置：锚定工程根、统一计时/跳过/子进程调用。"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def training_budget() -> int:
    """训练预算的**唯一真源**：configs/algo.yaml 的 ppo.total_epochs。

    各 run_XX 脚本此前各自硬编码 EPOCHS=1000，与 configs 里的值不一致，且注释还写着
    "与 configs/algo.yaml 一致"。预算一旦有两处定义就必然漂移，而等预算是方法间
    可比性的前提——主方法和消融拿到不同预算，比较就作废了。
    """
    from configs.config import load_config
    return int(load_config().get("ppo.total_epochs"))


def step(title):
    print("\n" + "=" * 72 + f"\n[RUN] {title}\n" + "=" * 72, flush=True)


def done(t0, *outputs):
    print(f"[OK] 用时 {time.time() - t0:.1f}s", flush=True)
    for p in outputs:
        p = Path(p)
        print(f"     产物 {p.resolve()} {'[已落盘]' if p.exists() else '[缺失!]'}", flush=True)


def skip_if_exists(path, what=""):
    p = Path(path)
    if p.exists():
        print(f"[SKIP] {what} 已完成，跳过：{p}", flush=True)
        return True
    return False


def run_py(entry, *args):
    head = [entry] if entry.endswith(".py") else ["-m", entry]
    cmd = [sys.executable] + head + [str(a) for a in args]
    print("[CMD] " + " ".join(cmd), flush=True)
    if subprocess.call(cmd, cwd=str(ROOT)) != 0:
        sys.exit(f"[FAIL] {entry} 非零退出，后续步骤已中止")


# 冒烟/探针类 run 只跑几个 epoch，用来验证链路是否跑得通，绝不能进评测：
# 它们会作为一个"方法"混进对比表与 Friedman 检验，还会多占一次多重比较校正的名额，
# 把真正的比较的 p 值推高。run 目录名以这些前缀开头的一律排除。
EXCLUDED_RUN_PREFIXES = ("smoke", "probe", "ph0", "debug", "tmp", "test")


def checkpoints(pattern="*_run*"):
    """自动发现已训练的 checkpoint，复现者不需要手填任何路径。"""
    found = sorted((ROOT / "result").glob(f"{pattern}/checkpoint_best.pt"))
    kept = [c for c in found
            if not c.parent.name.startswith(EXCLUDED_RUN_PREFIXES)]
    for c in found:
        if c not in kept:
            print(f"[SKIP] {c.parent.name}：冒烟/探针 run，不计入评测", flush=True)
    return kept
