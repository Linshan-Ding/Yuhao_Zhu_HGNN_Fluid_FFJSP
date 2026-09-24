"""按 configs/instance.yaml 的设计表生成固定评测档（grid / val / small / ood）。

每档带固定种子，逐算例的种子由 (档种子, 序号) 派生，重建结果逐位相同；算例文件与
index.csv 随论文发布，是复现基准。用法：
    python data/dataset.py                 # 重建全部档并整体重写 index.csv
    python data/dataset.py --tiers grid    # 只重建指定档，其余档的索引行保留
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import ROOT, load_config                            # noqa: E402
from data.generator import Instance, build_instance, save_instance_csv  # noqa: E402

INSTANCE_ROOT = ROOT / "data" / "instances"
INDEX_COLUMNS = [
    "instance_id", "tier", "path", "S", "R", "J", "M", "machines_per_stage",
    "DDT", "rho_target", "arrival_process", "E_dt", "Lambda", "W_bar", "p_bar",
    "rho_sys", "iota", "regime", "seed",
]


def _gap_for_target_rho(proc_range, stage_count, machines_per_stage, target_rho) -> float:
    """由目标系统负荷 rho_sys = Lambda * W_bar / |M| 反推平均到达间隔。"""
    mean_p = (float(proc_range[0]) + float(proc_range[1])) / 2.0
    w_bar = stage_count * mean_p
    n_machine = stage_count * machines_per_stage
    return w_bar / (max(float(target_rho), 1e-9) * n_machine)


def _index_row(inst: Instance, path: Path, rho_target: float, seed: int) -> Dict[str, object]:
    return {
        "instance_id": inst.instance_id, "tier": inst.tier,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "S": inst.order_count, "R": inst.product_count, "J": inst.stage_count,
        "M": inst.machine_count,
        "machines_per_stage": "-".join(str(m) for m in inst.machines_per_stage),
        "DDT": inst.meta.get("DDT"), "rho_target": rho_target,
        "arrival_process": inst.meta.get("arrival_process"),
        "E_dt": round(float(inst.meta.get("E_dt", 0.0)), 4),
        "Lambda": round(float(inst.meta.get("Lambda", 0.0)), 8),
        "W_bar": round(float(inst.meta.get("W_bar", 0.0)), 3),
        "p_bar": round(float(inst.meta.get("p_bar", 0.0)), 3),
        "rho_sys": round(float(inst.meta.get("rho_sys", 0.0)), 4),
        "iota": round(float(inst.meta.get("iota", 0.0)), 4),
        "regime": inst.meta.get("regime"), "seed": seed,
    }


class _Tier:
    """一档的构造器：逐算例派生种子，写文件，收集索引行。"""

    def __init__(self, name: str, base_seed: int, rows: List[Dict[str, object]]) -> None:
        self.name, self.base_seed, self.rows, self.k = name, int(base_seed), rows, 0
        for stale in (INSTANCE_ROOT / name).glob("*.csv"):
            stale.unlink()

    def emit(self, rho_target: float, **kwargs) -> None:
        seed = self.base_seed * 1000 + self.k
        self.k += 1
        inst = build_instance(np.random.default_rng(seed), tier=self.name, **kwargs)
        path = INSTANCE_ROOT / self.name / f"{inst.instance_id}.csv"
        save_instance_csv(inst, path)
        self.rows.append(_index_row(inst, path, rho_target, seed))


def _structure(pt, **override):
    J = int(override.get("stage_count", pt["stage_count"]))
    mps = int(override.get("machines_per_stage", pt["machines_per_stage"]))
    return {"product_count": int(override.get("product_count", pt["product_count"])),
            "stage_count": J, "machines_per_stage": [mps] * J,
            "proc_time_range": pt["proc_time_range"], "ddt_spread": pt.get("ddt_spread", (1.0, 1.0))}


def make_grid(cfg, rows):
    d, pt = cfg.get("design.grid"), cfg.get("param_table")
    tier = _Tier("grid", d["seed"], rows)
    for rho in d["rho_levels"]:
        for ddt in d["ddt_levels"]:
            for S in d["order_counts"]:
                st = _structure(pt)
                gap = _gap_for_target_rho(pt["proc_time_range"], st["stage_count"],
                                          int(pt["machines_per_stage"]), rho)
                tier.emit(float(rho), instance_id=f"grid_rho{str(rho).replace('.', 'p')}_DDT{ddt}_S{S}",
                          order_count=int(S), ddt=float(ddt), mean_interarrival=gap,
                          arrival_process="poisson", **st)


def make_val(cfg, rows):
    d, pt = cfg.get("design.val"), cfg.get("param_table")
    tier = _Tier("val", d["seed"], rows)
    for rho in d["rho_levels"]:
        for ddt in d["ddt_levels"]:
            st = _structure(pt)
            gap = _gap_for_target_rho(pt["proc_time_range"], st["stage_count"],
                                      int(pt["machines_per_stage"]), rho)
            tier.emit(float(rho), instance_id=f"val_rho{str(rho).replace('.', 'p')}_DDT{ddt}",
                      order_count=int(d["order_count"]), ddt=float(ddt), mean_interarrival=gap,
                      arrival_process="poisson", **st)


def make_small(cfg, rows):
    d, pt = cfg.get("design.small"), cfg.get("param_table")
    tier = _Tier("small", d["seed"], rows)
    st = _structure(pt)
    gap = _gap_for_target_rho(pt["proc_time_range"], st["stage_count"],
                              int(pt["machines_per_stage"]), d["target_rho_sys"])
    for S in d["order_counts"]:
        for ddt in d["ddt_levels"]:
            for k in range(int(d["instances_per_cell"])):
                tier.emit(float(d["target_rho_sys"]), instance_id=f"small_S{S}_DDT{ddt}_c{k + 1}",
                          order_count=int(S), ddt=float(ddt), mean_interarrival=gap,
                          arrival_process="poisson", **st)


def make_ood(cfg, rows):
    d, pt = cfg.get("design.ood"), cfg.get("param_table")
    tier = _Tier("ood", d["seed"], rows)
    for cond in d["conditions"]:
        st = _structure(pt, **{k: v for k, v in cond.items()
                               if k in ("stage_count", "machines_per_stage", "product_count")})
        gap = _gap_for_target_rho(pt["proc_time_range"], st["stage_count"],
                                  st["machines_per_stage"][0], d["target_rho_sys"])
        tier.emit(float(d["target_rho_sys"]), instance_id=f"ood_{cond['name']}",
                  order_count=int(cond.get("order_count", d["order_count_default"])),
                  ddt=float(d["ddt"]), mean_interarrival=gap,
                  arrival_process=str(cond.get("arrival_process", "poisson")), **st)


TIER_BUILDERS = {"grid": make_grid, "val": make_val, "small": make_small, "ood": make_ood}


def make_eval_instances(tiers: List[str] | None = None) -> Path:
    cfg = load_config()
    tiers = list(tiers or TIER_BUILDERS)
    index_path = INSTANCE_ROOT / "index.csv"
    kept: List[Dict[str, object]] = []
    if index_path.exists() and set(tiers) != set(TIER_BUILDERS):
        with index_path.open("r", encoding="utf-8") as handle:
            kept = [r for r in csv.DictReader(handle) if r["tier"] not in tiers]
    rows: List[Dict[str, object]] = []
    for tier in tiers:
        before = len(rows)
        TIER_BUILDERS[tier](cfg, rows)
        print(f"[OK] tier '{tier}' 生成 {len(rows) - before} 个算例", flush=True)
    all_rows = kept + rows
    all_rows.sort(key=lambda r: (str(r["tier"]), str(r["instance_id"])))
    INSTANCE_ROOT.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[OK] index.csv 共 {len(all_rows)} 行 -> {index_path}", flush=True)
    return index_path


def read_index(tier: str | None = None) -> List[Dict[str, str]]:
    index_path = INSTANCE_ROOT / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError("data/instances/index.csv 不存在，请先运行 scripts/run_01_prepare_data.py")
    with index_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [r for r in rows if tier is None or r["tier"] == tier]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", nargs="+", choices=list(TIER_BUILDERS), default=None)
    make_eval_instances(parser.parse_args().tiers)
