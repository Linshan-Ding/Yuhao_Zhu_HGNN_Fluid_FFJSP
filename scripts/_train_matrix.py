"""训练矩阵的公共启动器：按 RUN_SPECS 训练一组 run，已训满预算的跳过。"""
import argparse
import time

from _bootstrap import ROOT, RUN_SPECS, done, run_complete, run_py, run_seed, step, training_budget


def train_tags(tags, title):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--only", nargs="+", choices=tags, default=tags)
    parser.add_argument("--runs", type=int, default=None, help="每个配置的独立 run 数（默认 runtime.n_runs）")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    from configs.config import load_config
    n_runs = args.runs or int(load_config().get("runtime.n_runs", 5))
    budget = args.total_steps or training_budget()
    for tag in args.only:
        method, variant, overlays = RUN_SPECS[tag]
        for i in range(1, n_runs + 1):
            name = f"{tag}_run{i}"
            run_dir = ROOT / "result" / name
            if run_complete(run_dir, budget):
                print(f"[SKIP] {name} 已训满 {budget} 步", flush=True)
                continue
            t0 = time.time()
            step(f"训练 {variant}（{name}，seed={run_seed(i)}，预算 {budget} 步）")
            extra = ["--total-steps", budget, "--seed", run_seed(i)]
            if args.workers:
                extra += ["--workers", args.workers]
            if (run_dir / "checkpoint_last.pt").exists():
                extra.append("--resume")
            run_py("train.py", "--config", *overlays, "--run-name", name, *extra)
            done(t0, run_dir / "checkpoint_best.pt", run_dir / "log.csv")
