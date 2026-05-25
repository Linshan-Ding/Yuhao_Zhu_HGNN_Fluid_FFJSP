import argparse
import copy
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.PPO_model import PPO
from agent.sci_rollout import collect_vectorized_rollouts, resolve_instance_files, select_instance
from agent.visualization import TrainingVisualizer


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_data_dir(path_value: str) -> Path:
    data_dir = Path(path_value)
    if not data_dir.is_absolute():
        data_dir = ROOT_DIR / data_dir
    return data_dir


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_metrics(metrics_path: Path, row: dict):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    exists = metrics_path.exists()
    write_header = (not exists) or metrics_path.stat().st_size == 0
    fieldnames = list(row.keys())
    if exists:
        with metrics_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_fields = [field for field in (reader.fieldnames or []) if field is not None]
            if existing_fields and existing_fields != fieldnames:
                rows = []
                for old_row in reader:
                    old_row.pop(None, None)
                    rows.append({name: old_row.get(name, "") for name in existing_fields})
                fieldnames = existing_fields + [name for name in fieldnames if name not in existing_fields]
                with metrics_path.open("w", newline="", encoding="utf-8") as rewrite_file:
                    writer = csv.DictWriter(rewrite_file, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                write_header = False
            elif existing_fields:
                fieldnames = existing_fields
    with metrics_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_instances(values):
    if not values:
        return None
    instances = []
    for value in values:
        instances.extend(item.strip() for item in value.split(",") if item.strip())
    return instances


def build_model_paras(config: dict, device, train_paras: dict | None = None):
    model_paras = dict(config["model_paras"])
    train_paras = train_paras or config.get("train_paras", {})
    model_paras["device"] = device
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    model_paras["use_sparse_attention"] = bool(train_paras.get("use_sparse_attention", False))
    model_paras["sparse_attention_scope"] = str(train_paras.get("sparse_attention_scope", "pair"))
    model_paras["sparse_attention_heads"] = int(train_paras.get("sparse_attention_heads", 1))
    model_paras["sparse_attention_dropout"] = float(train_paras.get("sparse_attention_dropout", 0.0))
    return model_paras


def resolve_torch_device(name: str | None, fallback: str = "cpu") -> torch.device:
    requested = str(name or fallback)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[Device] CUDA requested as '{requested}' but is unavailable; falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def prepare_rollout_policy(model: PPO, rollout_device: torch.device, update_device: torch.device):
    if rollout_device == update_device:
        model.policy_old.eval()
        return model.policy_old
    policy = copy.deepcopy(model.policy_old).to(rollout_device)
    policy.device = rollout_device
    policy.eval()
    return policy


def save_checkpoint(path: Path, model: PPO, iteration: int, best_fulfillment_rate: float,
                    train_paras: dict, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iteration": int(iteration),
        "best_fulfillment_rate": float(best_fulfillment_rate),
        "policy": model.policy.state_dict(),
        "policy_old": model.policy_old.state_dict(),
        "optimizer": model.optimizer.state_dict(),
        "train_paras": train_paras,
        "config": config,
    }, path)


def load_checkpoint(path: Path, model: PPO, device: torch.device) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.policy.load_state_dict(checkpoint["policy"])
    model.policy_old.load_state_dict(checkpoint.get("policy_old", checkpoint["policy"]))
    if "optimizer" in checkpoint:
        model.optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("iteration", -1)) + 1, float(checkpoint.get("best_fulfillment_rate", float("-inf")))


def gpu_temperature() -> float:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2,
        )
        first_line = output.strip().splitlines()[0]
        return float(first_line)
    except Exception:
        return 0.0


def resource_snapshot(update_device: torch.device) -> dict:
    cpu_memory_mb = 0.0
    try:
        import psutil

        cpu_memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        pass

    gpu_memory_mb = 0.0
    gpu_temp = 0.0
    if update_device.type == "cuda" and torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.max_memory_allocated(update_device) / (1024 ** 2)
        gpu_temp = gpu_temperature()
    return {
        "cpu_memory_mb": cpu_memory_mb,
        "gpu_memory_mb": gpu_memory_mb,
        "gpu_temperature": gpu_temp,
    }


def apply_lr_schedule(model: PPO, train_paras: dict, iteration: int, max_iterations: int) -> float:
    schedule = str(train_paras.get("lr_schedule", "none")).lower()
    base_lr = float(train_paras.get("lr", 3e-4))
    min_lr = float(train_paras.get("min_lr", base_lr))
    if schedule == "linear" and max_iterations > 1:
        progress = min(max(iteration / (max_iterations - 1), 0.0), 1.0)
        lr = base_lr + progress * (min_lr - base_lr)
    elif schedule in {"none", ""}:
        lr = base_lr
    else:
        raise ValueError("lr_schedule must be 'none' or 'linear'")
    model.set_learning_rate(lr)
    return lr


def main():
    parser = argparse.ArgumentParser(description="Parallel HGNN-PPO trainer")
    parser.add_argument("--config", default=str(ROOT_DIR / "data" / "config.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rollout-device", default=None)
    parser.add_argument("--update-device", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--instance-selection", choices=["fixed", "round_robin", "random"], default=None)
    parser.add_argument("--resume", choices=["auto", "none"], default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    train_paras = dict(config["train_paras"])
    if args.max_iterations is not None:
        train_paras["max_iterations"] = args.max_iterations
    if args.instance_selection is not None:
        train_paras["instance_selection"] = args.instance_selection
    if args.rollout_device is not None:
        train_paras["rollout_device"] = args.rollout_device
    if args.update_device is not None:
        train_paras["update_device"] = args.update_device
    if args.resume is not None:
        train_paras["resume"] = args.resume
    set_seed(args.seed)

    update_device_name = args.device or train_paras.get("update_device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    rollout_device_name = train_paras.get("rollout_device", "cpu")
    update_device = resolve_torch_device(update_device_name, fallback="cpu")
    rollout_device = resolve_torch_device(rollout_device_name, fallback="cpu")
    data_dir = resolve_data_dir(args.data_dir or train_paras.get("train_data_dir", "data/instance/competition"))
    patterns = train_paras.get("train_patterns", ["*.txt"])
    if isinstance(patterns, str):
        patterns = [patterns]
    requested_instances = parse_instances(args.instances)
    if requested_instances is None:
        config_instances = train_paras.get("train_instances") or None
        requested_instances = parse_instances([config_instances]) if isinstance(config_instances, str) else config_instances
    instance_files = resolve_instance_files(str(data_dir), patterns, requested_instances)

    model = PPO(build_model_paras(config, update_device, train_paras), train_paras)
    visualizer = TrainingVisualizer(train_paras)
    result_dir = ROOT_DIR / "result"
    metrics_path = result_dir / "train_metrics.csv"
    last_checkpoint_path = result_dir / "last_checkpoint.pt"
    best_checkpoint_path = result_dir / "best_checkpoint.pt"
    best_fulfillment_rate = float("-inf")
    trajectory_base = 0
    start_iteration = 0
    if train_paras.get("resume", "auto") == "auto" and last_checkpoint_path.exists():
        try:
            start_iteration, best_fulfillment_rate = load_checkpoint(last_checkpoint_path, model, update_device)
            print(f"[Checkpoint] Resumed from iteration {start_iteration}; best fulfillment={best_fulfillment_rate:.6f}")
        except Exception as exc:
            print(f"[Checkpoint] Could not resume from {last_checkpoint_path}: {exc}. Starting fresh.")

    max_iterations = int(train_paras.get("max_iterations", 1000))
    try:
        for iteration in range(start_iteration, max_iterations):
            if update_device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(update_device)
            start_time = time.time()
            learning_rate = apply_lr_schedule(model, train_paras, iteration, max_iterations)
            instance_selection = train_paras.get("instance_selection", "fixed")
            instance_index, selected_instance = select_instance(instance_files, instance_selection, iteration)
            rollout_start_time = time.time()
            rollout_policy = prepare_rollout_policy(model, rollout_device, update_device)
            rollout = collect_vectorized_rollouts(rollout_policy, selected_instance, train_paras, trajectory_base)
            rollout_seconds = time.time() - rollout_start_time
            num_envs = int(train_paras.get("num_envs", train_paras.get("num_workers", 4)))
            trajectory_base += num_envs
            rollout.minibatch_size = int(train_paras.get("minibatch_size", 512))
            update_start_time = time.time()
            rollout.compute_returns_and_advantages(
                gamma=float(train_paras.get("gamma", 1.0)),
                gae_lambda=float(train_paras.get("gae_lambda", 0.95)),
            )
            update_metrics = model.update_rollout(rollout)
            update_seconds = time.time() - update_start_time
            elapsed = time.time() - start_time
            resources = resource_snapshot(update_device)
            total_reward = float(sum(rollout.rewards))
            steps_per_second = len(rollout) / max(elapsed, 1e-8)
            mean_return = float(np.mean(rollout.returns)) if rollout.returns else 0.0
            loss = float(update_metrics.get("loss", 0.0))
            fulfillment_metrics = rollout.fulfillment_summary()
            episode_metrics = rollout.episode_summary()
            row = {
                "iteration": iteration,
                "instance_id": instance_index + 1,
                "instance_name": selected_instance.name,
                "instance_selection": instance_selection,
                "transitions": len(rollout),
                "episode_count": episode_metrics["episode_count"],
                "episode_transition_mean": episode_metrics["episode_transition_mean"],
                "episode_transition_std": episode_metrics["episode_transition_std"],
                "episode_seconds_mean": episode_metrics["episode_seconds_mean"],
                "episode_seconds_std": episode_metrics["episode_seconds_std"],
                "reward": total_reward,
                "mean_return": mean_return,
                "fulfillment_rate_mean": fulfillment_metrics["fulfillment_rate_mean"],
                "fulfillment_rate_std": fulfillment_metrics["fulfillment_rate_std"],
                "fulfillment_rate_min": fulfillment_metrics["fulfillment_rate_min"],
                "fulfillment_rate_max": fulfillment_metrics["fulfillment_rate_max"],
                "loss": loss,
                "policy_loss": float(update_metrics.get("policy_loss", 0.0)),
                "value_loss": float(update_metrics.get("value_loss", 0.0)),
                "entropy": float(update_metrics.get("entropy", 0.0)),
                "approx_kl": float(update_metrics.get("approx_kl", 0.0)),
                "clip_fraction": float(update_metrics.get("clip_fraction", 0.0)),
                "learning_rate": learning_rate,
                "discard_count": rollout.discard_count,
                "completion_count": rollout.completion_count,
                "steps_per_second": steps_per_second,
                "rollout_seconds": rollout_seconds,
                "update_seconds": update_seconds,
                "elapsed_seconds": elapsed,
                "fluid_solve_count": rollout.fluid_solve_count,
                "fluid_solve_seconds": rollout.fluid_solve_seconds,
                "fluid_cache_hit_count": rollout.fluid_cache_hit_count,
                "fluid_fallback_count": rollout.fluid_fallback_count,
                "sparse_attention_enabled": bool(train_paras.get("use_sparse_attention", False)),
                "sparse_attention_scope": str(train_paras.get("sparse_attention_scope", "pair")),
                "policy_forward_count": rollout.policy_forward_count,
                "policy_forward_seconds": rollout.policy_forward_seconds,
                "avg_action_tokens": rollout.avg_action_tokens,
                "max_action_tokens": rollout.max_action_tokens,
                "padding_ratio": rollout.padding_ratio,
                "actor_batch_size_mean": rollout.actor_batch_size_mean,
                "policy_act_seconds": rollout.policy_forward_seconds,
                "simulator_step_seconds": rollout.simulator_step_seconds,
                "obs_build_seconds": rollout.obs_build_seconds,
                "rollout_device": str(rollout_device),
                "update_device": str(update_device),
                "cpu_memory_mb": resources["cpu_memory_mb"],
                "gpu_memory_mb": resources["gpu_memory_mb"],
                "gpu_temperature": resources["gpu_temperature"],
            }
            write_metrics(metrics_path, row)
            visualizer.log(iteration, row)
            print(row)

            if fulfillment_metrics["fulfillment_rate_mean"] >= best_fulfillment_rate:
                best_fulfillment_rate = fulfillment_metrics["fulfillment_rate_mean"]
                save_checkpoint(best_checkpoint_path, model, iteration, best_fulfillment_rate, train_paras, config)
                torch.save(model.policy.state_dict(), result_dir / "ppo_policy_model.pt")
            checkpoint_interval = max(int(train_paras.get("checkpoint_interval", 1)), 1)
            if iteration % checkpoint_interval == 0:
                save_checkpoint(last_checkpoint_path, model, iteration, best_fulfillment_rate, train_paras, config)
    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
