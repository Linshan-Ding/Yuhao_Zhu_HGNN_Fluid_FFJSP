"""生成四档固定评测算例（grid / val / small / ood）。带固定种子，重建结果逐位相同。"""
import time

from _bootstrap import ROOT, done, run_py, step

t0 = time.time()
step("生成评测算例（grid 36 / val 6 / small 24 / ood 7，固定种子）")
run_py("data/dataset.py")
done(t0, ROOT / "data" / "instances" / "index.csv")
