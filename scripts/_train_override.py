"""训练包装器：允许用 --set key=value 覆盖单个配置项（仅供 run_09 的网格使用）。"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from _bootstrap import ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--set", action="append", default=[])
args = parser.parse_args()

overlay = {}
for item in args.set:
    key, value = item.split("=", 1)
    node = overlay
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(value)

tmp = Path(tempfile.mkdtemp()) / "override.yaml"
tmp.write_text(yaml.safe_dump(overlay, allow_unicode=True), encoding="utf-8")
code = subprocess.call([sys.executable, "train.py", "--config", str(tmp),
                        "--run-name", args.run_name, "--epochs", str(args.epochs)],
                       cwd=str(ROOT))
sys.exit(code)
