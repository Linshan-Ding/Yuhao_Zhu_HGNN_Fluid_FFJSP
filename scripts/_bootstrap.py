"""所有 run_XX 脚本的公共前置：锚定工程根、统一计时/跳过/子进程调用。"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


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


def checkpoints(pattern="*_run*"):
    """自动发现已训练的 checkpoint，复现者不需要手填任何路径。"""
    return sorted((ROOT / "result").glob(f"{pattern}/checkpoint_best.pt"))
