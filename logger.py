"""日志：CSV 持久化（论文数据）+ 可选 visdom 实时曲线（训练时看）。"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Dict, List

from configs.config import ROOT


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


class CsvLogger:
    """一次 run 的落盘器：log.csv + config_snapshot.yaml + commit.txt。"""

    def __init__(self, run_dir: str | Path, columns: List[str]) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "log.csv"
        self.columns = columns
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=columns).writeheader()
        (self.dir / "commit.txt").write_text(git_commit(), encoding="utf-8")

    def log(self, row: Dict[str, object]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.columns).writerow(
                {k: row.get(k, "") for k in self.columns})


class VisdomLogger:
    """可选实时监督。服务未开时静默降级，绝不影响训练。"""

    def __init__(self, enabled: bool = True, env_name: str = "FSHGRL") -> None:
        self.vis = None
        if not enabled:
            return
        try:
            import visdom
            self.vis = visdom.Visdom(env=env_name, raise_exceptions=False)
            if not self.vis.check_connection():
                self.vis = None
        except Exception:
            self.vis = None

    def line(self, win: str, x: float, y: float) -> None:
        if self.vis is None:
            return
        try:
            self.vis.line(X=[x], Y=[y], win=win, update="append",
                          opts={"title": win, "xlabel": "iteration"})
        except Exception:
            self.vis = None


def append_rows(path: str | Path, rows: List[Dict[str, object]], columns: List[str]) -> Path:
    """把若干行追加进结果 CSV（不存在则建表头）。所有实验结果都走这一个出口。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
    return path
