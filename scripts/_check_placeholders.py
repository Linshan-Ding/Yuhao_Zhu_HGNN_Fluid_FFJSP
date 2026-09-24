"""核对论文里的 \\PH{} 键集合与 run_09 的 SOURCES 键集合逐一相等。

    python scripts/_check_placeholders.py [--paper-dir ../Yuhao_Zhu_FFJSP_Order_GNN_Fluid_model]

只读论文的 main.tex 与 sections/*.tex（不需要任何实验数据），差集非空即以非零码退出。run_00 也会调用。
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ROOT  # noqa: E402

import run_09_fill_placeholders as r09  # noqa: E402


def paper_keys(paper: Path):
    keys = set()
    for path in [paper / "main.tex"] + sorted((paper / "sections").glob("*.tex")):
        for line in path.read_text(encoding="utf-8").splitlines():
            code = re.split(r"(?<!\\)%", line, 1)[0]          # 去掉 LaTeX 注释
            keys |= set(re.findall(r"\\PH\{([^}]+)\}", code))
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", default="../Yuhao_Zhu_FFJSP_Order_GNN_Fluid_model")
    args = parser.parse_args()
    paper = Path(args.paper_dir)
    if not paper.is_absolute():
        paper = (ROOT / paper).resolve()
    if not (paper / "main.tex").exists():
        raise SystemExit(f"[FAIL] 找不到论文目录 {paper}")
    src = set(r09.sources(r09.Data()))
    tex = paper_keys(paper)
    only_paper, only_src = sorted(tex - src), sorted(src - tex)
    print(f"  论文 \\PH 键 {len(tex)} 个，SOURCES 键 {len(src)} 个", flush=True)
    if only_paper:
        print("  [FAIL] 论文用了但 run_09 不产出：" + ", ".join(only_paper))
    if only_src:
        print("  [FAIL] run_09 产出但论文没用：" + ", ".join(only_src))
    if only_paper or only_src:
        raise SystemExit(1)
    print("[OK] 论文占位符集合与 SOURCES 逐一相等", flush=True)


if __name__ == "__main__":
    main()
