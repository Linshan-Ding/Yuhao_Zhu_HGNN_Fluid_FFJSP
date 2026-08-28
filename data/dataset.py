"""按 docs/experiment-spec.md §6A 的算例设计表逐单元生成固定评测/验证算例。

一次生成永久固定、随论文发布——复现基准是这些算例文件本身 + 多次独立 run，不是随机种子。
重复运行会跳过已存在的档位。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import ROOT, load_config          # noqa: E402
from data.generator import Instance, build_instance, save_instance_csv  # noqa: E402

INSTANCE_ROOT = ROOT / "data" / "instances"
INDEX_COLUMNS = [
    "instance_id", "tier", "path", "S", "R", "J", "M", "machines_per_stage",
    "DDT", "arrival_process", "E_dt", "Lambda", "W_bar", "p_bar",
    "rho_sys", "iota", "regime",
]


def _index_row(inst: Instance, path: Path) -> Dict[str, object]:
    return {
        "instance_id": inst.instance_id, "tier": inst.tier,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "S": inst.order_count, "R": inst.product_count, "J": inst.stage_count,
        "M": inst.machine_count,
        "machines_per_stage": "-".join(str(m) for m in inst.machines_per_stage),
        "DDT": inst.meta.get("DDT"), "arrival_process": inst.meta.get("arrival_process"),
        "E_dt": round(float(inst.meta.get("E_dt", 0.0)), 4),
        "Lambda": round(float(inst.meta.get("Lambda", 0.0)), 8),
        "W_bar": round(float(inst.meta.get("W_bar", 0.0)), 3),
        "p_bar": round(float(inst.meta.get("p_bar", 0.0)), 3),
        "rho_sys": round(float(inst.meta.get("rho_sys", 0.0)), 4),
        "iota": round(float(inst.meta.get("iota", 0.0)), 4),
        "regime": inst.meta.get("regime"),
    }


def _emit(inst: Instance, rows: List[Dict[str, object]]) -> None:
    path = INSTANCE_ROOT / inst.tier / f"{inst.instance_id}.csv"
    save_instance_csv(inst, path)
    rows.append(_index_row(inst, path))


def _gap_for_target_rho(proc_range, stage_count, machines_per_stage, target_rho) -> float:
    """由目标系统负荷 rho_sys = Lambda * W_bar / |M| 反推平均到达间隔。"""
    mean_p = (float(proc_range[0]) + float(proc_range[1])) / 2.0
    w_bar = stage_count * mean_p
    n_machine = stage_count * machines_per_stage
    return w_bar / (max(float(target_rho), 1e-9) * n_machine)


def make_small(cfg, rng, rows):
    d = cfg.get("design.small")
    pt = cfg.get("param_table")
    proc_range = d.get("proc_time_range", pt["proc_time_range"])
    gap = _gap_for_target_rho(proc_range, int(d["stage_count"]),
                              int(d["machines_per_stage"]), d.get("target_rho_sys", 1.3))
    for S in d["order_counts"]:
        for ddt in d["ddt_levels"]:
            for k in range(int(d["instances_per_cell"])):
                iid = f"small_S{S}_DDT{ddt}_c{k+1}"
                _emit(build_instance(
                    rng, instance_id=iid, tier="small",
                    product_count=int(d["product_count"]), stage_count=int(d["stage_count"]),
                    machines_per_stage=[int(d["machines_per_stage"])] * int(d["stage_count"]),
                    order_count=int(S), proc_time_range=proc_range,
                    ddt=float(ddt), mean_interarrival=gap,
                    arrival_process="poisson"), rows)


def make_main(cfg, rng, rows):
    d = cfg.get("design.main")
    pt = cfg.get("param_table")
    J = int(pt["stage_count"])
    for ddt in d["ddt_levels"]:
        for S in d["order_counts"]:
            iid = f"main_DDT{ddt}_S{S}"
            _emit(build_instance(
                rng, instance_id=iid, tier="main",
                product_count=int(pt["product_count"]), stage_count=J,
                machines_per_stage=[int(pt["machines_per_stage"])] * J,
                order_count=int(S), proc_time_range=pt["proc_time_range"],
                ddt=float(ddt),
                mean_interarrival=float(sum(pt["interarrival_range"])) / 2.0,
                arrival_process="poisson"), rows)


def make_arrival(cfg, rng, rows):
    d = cfg.get("design.arrival")
    pt = cfg.get("param_table")
    J = int(pt["stage_count"])
    for gap in d["mean_interarrival"]:
        for proc in d["processes"]:
            iid = f"arr_dt{str(gap).replace('.', 'p')}_{proc}"
            _emit(build_instance(
                rng, instance_id=iid, tier="arrival",
                product_count=int(pt["product_count"]), stage_count=J,
                machines_per_stage=[int(pt["machines_per_stage"])] * J,
                order_count=int(d["order_count"]), proc_time_range=pt["proc_time_range"],
                ddt=float(d["ddt"]), mean_interarrival=float(gap),
                arrival_process=proc), rows)


def make_ood(cfg, rng, rows):
    d = cfg.get("design.ood")
    pt = cfg.get("param_table")
    for cond in d["conditions"]:
        J = int(cond.get("stage_count", pt["stage_count"]))
        mps = int(cond.get("machines_per_stage", pt["machines_per_stage"]))
        _emit(build_instance(
            rng, instance_id=f"ood_{cond['name']}", tier="ood",
            product_count=int(cond.get("product_count", pt["product_count"])),
            stage_count=J, machines_per_stage=[mps] * J,
            order_count=int(cond.get("order_count", d["order_count_default"])),
            proc_time_range=pt["proc_time_range"], ddt=float(d["ddt"]),
            mean_interarrival=float(sum(pt["interarrival_range"])) / 2.0,
            arrival_process="poisson"), rows)


def make_val(cfg, rng, rows):
    d = cfg.get("design.val")
    pt = cfg.get("param_table")
    J = int(pt["stage_count"])
    for k in range(int(d["count"])):
        _emit(build_instance(
            rng, instance_id=f"val_{k+1}", tier="val",
            product_count=int(pt["product_count"]), stage_count=J,
            machines_per_stage=[int(pt["machines_per_stage"])] * J,
            order_count=int(d["order_count"]), proc_time_range=pt["proc_time_range"],
            ddt=float(d["ddt"]),
            mean_interarrival=float(sum(pt["interarrival_range"])) / 2.0,
            arrival_process="poisson"), rows)


def make_case3d(cfg, rng, rows):
    d = cfg.get("design.case3d")
    mps = [int(m) for m in d["machines_per_stage"]]
    J = len(mps)
    for ddt in d["ddt_levels"]:
        for S in d["order_counts"]:
            _emit(build_instance(
                rng, instance_id=f"case3d_DDT{ddt}_S{S}", tier="case3d",
                product_count=int(d["product_count"]), stage_count=J,
                machines_per_stage=mps, order_count=int(S),
                proc_time_range=d["proc_time_range"], ddt=float(ddt),
                mean_interarrival=float(ddt) / 6.0, arrival_process="poisson",
                eligibility_prob=0.8), rows)


TIER_BUILDERS = {
    "small": make_small, "main": make_main, "arrival": make_arrival,
    "ood": make_ood, "val": make_val, "case3d": make_case3d,
}


def make_eval_instances(tiers: List[str] | None = None, force: bool = False) -> Path:
    cfg = load_config()
    rng = np.random.default_rng()
    tiers = tiers or list(TIER_BUILDERS)
    rows: List[Dict[str, object]] = []

    index_path = INSTANCE_ROOT / "index.csv"
    existing: List[Dict[str, object]] = []
    if index_path.exists() and not force:
        with index_path.open("r", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))

    kept = [r for r in existing if r["tier"] not in tiers]
    for tier in tiers:
        already = [r for r in existing if r["tier"] == tier]
        if already and not force:
            print(f"[SKIP] tier '{tier}' 已有 {len(already)} 个算例")
            kept.extend(already)
            continue
        before = len(rows)
        TIER_BUILDERS[tier](cfg, rng, rows)
        print(f"[OK] tier '{tier}' 生成 {len(rows) - before} 个算例")

    all_rows = kept + rows
    all_rows.sort(key=lambda r: (str(r["tier"]), str(r["instance_id"])))
    INSTANCE_ROOT.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[OK] index.csv 共 {len(all_rows)} 行 -> {index_path}")
    return index_path


def read_index(tier: str | None = None) -> List[Dict[str, str]]:
    index_path = INSTANCE_ROOT / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError("data/instances/index.csv 不存在，请先运行 scripts/run_01_prepare_data.py")
    with index_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [r for r in rows if tier is None or r["tier"] == tier]


if __name__ == "__main__":
    make_eval_instances(force="--force" in sys.argv)
