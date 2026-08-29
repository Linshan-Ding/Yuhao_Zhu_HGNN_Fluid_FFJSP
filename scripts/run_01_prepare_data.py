"""生成六档固定评测/验证算例 + index.csv（含逐算例负荷指标）。

一次生成永久固定、随论文发布——复现基准是这些算例文件本身，不是随机种子。
已存在的档位自动跳过；要重建先删 data/instances/。
"""
import time

from _bootstrap import ROOT, done, run_py, step

t0 = time.time()
step("生成固定评测/验证算例（small / main / arrival / ood / val / case3d）")
run_py("data/dataset.py")
done(t0, ROOT / "data" / "instances" / "index.csv")
