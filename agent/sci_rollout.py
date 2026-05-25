from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

from agent.training_simulator import TrainingSimulator, parse_instance_file


def scan_instance_files(data_dir: str, patterns: Iterable[str]) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        if "isntance" in str(root):
            raise FileNotFoundError(
                f"Training data directory '{root}' does not exist. "
                "Use 'data/instance/competition' instead of 'data/isntance/competition'."
            )
        raise FileNotFoundError(f"Training data directory '{root}' does not exist.")

    files = []
    for pattern in patterns:
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    files = sorted({
        path.resolve()
        for path in files
        if not path.name.startswith("._") and path.name != "Introduction.txt"
    })
    if not files:
        raise FileNotFoundError(f"No training instance files matched {list(patterns)} under '{root}'.")
    return files


def resolve_instance_files(data_dir: str, patterns: Iterable[str],
                           instance_names: Sequence[str] | None = None) -> List[Path]:
    all_files = scan_instance_files(data_dir, ["*.txt"] if instance_names else patterns)
    if not instance_names:
        return all_files

    by_name = {path.name: path for path in all_files}
    selected = []
    missing = []
    for name in instance_names:
        key = Path(name).name
        if key in by_name:
            selected.append(by_name[key])
        else:
            missing.append(name)
    if missing:
        available = ", ".join(sorted(by_name.keys()))
        raise FileNotFoundError(
            f"Training instances not found under '{data_dir}': {missing}. "
            f"Available instances: {available}"
        )
    return selected


def select_instance(instance_files: List[Path], strategy: str, index: int) -> tuple[int, Path]:
    if not instance_files:
        raise ValueError("No training instances are available.")
    strategy = (strategy or "fixed").lower()
    if strategy == "random":
        selected_index = random.randrange(len(instance_files))
        return selected_index, instance_files[selected_index]
    if strategy in {"fixed", "round_robin"}:
        selected_index = index % len(instance_files)
        return selected_index, instance_files[selected_index]
    raise ValueError("instance_selection must be one of: fixed, round_robin, random")


@dataclass
class RolloutBuffer:
    obs: List[dict] = field(default_factory=list)
    action_indexes: List[int] = field(default_factory=list)
    env_actions: List[tuple] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    trajectory_ids: List[int] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    minibatch_size: int = 512
    max_minibatch_tokens: int = 0
    discard_count: int = 0
    completion_count: int = 0
    fulfillment_rates: List[float] = field(default_factory=list)
    fluid_solve_count: int = 0
    fluid_solve_seconds: float = 0.0
    fluid_cache_hit_count: int = 0
    fluid_fallback_count: int = 0
    policy_forward_count: int = 0
    policy_forward_seconds: float = 0.0
    avg_action_tokens: float = 0.0
    max_action_tokens: float = 0.0
    padding_ratio: float = 0.0
    actor_batch_size_mean: float = 0.0
    episode_transition_counts: List[int] = field(default_factory=list)
    episode_seconds: List[float] = field(default_factory=list)
    obs_build_seconds: float = 0.0
    simulator_step_seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.rewards)

    def add(self, obs, action_index, env_action, log_prob, value, reward, done, trajectory_id):
        self.obs.append(obs)
        self.action_indexes.append(int(action_index))
        self.env_actions.append((int(env_action[0]), int(env_action[1])))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.trajectory_ids.append(int(trajectory_id))

    def fulfillment_summary(self) -> dict:
        if not self.fulfillment_rates:
            return {
                "fulfillment_rate_mean": 0.0,
                "fulfillment_rate_std": 0.0,
                "fulfillment_rate_min": 0.0,
                "fulfillment_rate_max": 0.0,
            }
        values = torch.tensor(self.fulfillment_rates, dtype=torch.float32)
        return {
            "fulfillment_rate_mean": float(values.mean().item()),
            "fulfillment_rate_std": float(values.std(unbiased=False).item()),
            "fulfillment_rate_min": float(values.min().item()),
            "fulfillment_rate_max": float(values.max().item()),
        }

    def episode_summary(self) -> dict:
        transition_values = torch.tensor(self.episode_transition_counts or [0], dtype=torch.float32)
        second_values = torch.tensor(self.episode_seconds or [0.0], dtype=torch.float32)
        return {
            "episode_count": len(self.episode_transition_counts),
            "episode_transition_mean": float(transition_values.mean().item()),
            "episode_transition_std": float(transition_values.std(unbiased=False).item()),
            "episode_seconds_mean": float(second_values.mean().item()),
            "episode_seconds_std": float(second_values.std(unbiased=False).item()),
        }

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float):
        self.returns = [0.0 for _ in self.rewards]
        self.advantages = [0.0 for _ in self.rewards]
        by_trajectory = {}
        for index, trajectory_id in enumerate(self.trajectory_ids):
            by_trajectory.setdefault(trajectory_id, []).append(index)

        for indices in by_trajectory.values():
            gae = 0.0
            next_value = 0.0
            for index in reversed(indices):
                non_terminal = 0.0 if self.dones[index] else 1.0
                delta = self.rewards[index] + gamma * next_value * non_terminal - self.values[index]
                gae = delta + gamma * gae_lambda * non_terminal * gae
                self.advantages[index] = gae
                self.returns[index] = gae + self.values[index]
                next_value = self.values[index]

    def iter_minibatches(self, shuffle: bool = True):
        buckets = {}
        for index, obs in enumerate(self.obs):
            key = (
                int(obs.get("num_tasks", obs["raw_opes"].size(0))),
                int(obs.get("num_machines", obs["raw_mas"].size(0))),
                int(obs.get("max_orders", obs["due_dates"].size(1))),
            )
            buckets.setdefault(key, []).append(index)

        bucket_keys = list(buckets.keys())
        if shuffle:
            random.shuffle(bucket_keys)

        for key in bucket_keys:
            indices = buckets[key]
            if shuffle:
                random.shuffle(indices)
            token_per_sample = max(key[0] * key[1] * key[2], 1)
            token_limit = int(self.max_minibatch_tokens or 0)
            if token_limit > 0:
                batch_size = max(1, min(self.minibatch_size, token_limit // token_per_sample))
            else:
                batch_size = self.minibatch_size
            for start in range(0, len(indices), batch_size):
                yield indices[start:start + batch_size]


def _mark_last_transition_terminal(buffer: RolloutBuffer, trajectory_id: int):
    for index in range(len(buffer.dones) - 1, -1, -1):
        if buffer.trajectory_ids[index] == trajectory_id:
            buffer.dones[index] = True
            return


def collect_vectorized_rollouts(policy, instance_path: Path, train_paras: dict,
                                base_trajectory_id: int = 0) -> RolloutBuffer:
    spec = parse_instance_file(instance_path)
    num_envs = int(train_paras.get("num_envs", train_paras.get("num_workers", 4)))
    buffer = RolloutBuffer(minibatch_size=int(train_paras.get("minibatch_size", 512)))
    buffer.max_minibatch_tokens = int(train_paras.get("max_minibatch_tokens", 0))
    simulators = [TrainingSimulator(spec, train_paras) for _ in range(num_envs)]
    trajectory_ids = [base_trajectory_id + index for index in range(num_envs)]
    episode_start_times = [time.perf_counter() for _ in range(num_envs)]
    finished = [False for _ in range(num_envs)]
    policy_forward_count = 0
    policy_forward_seconds = 0.0
    total_requests = 0
    action_tokens_sum = 0.0
    max_action_tokens = 0.0
    padding_ratio_sum = 0.0

    while not all(finished):
        active_indices = [
            index for index, simulator in enumerate(simulators)
            if not finished[index] and not simulator.done and simulator.has_decision()
        ]
        if not active_indices:
            for index, simulator in enumerate(simulators):
                if not finished[index] and simulator.done:
                    finished[index] = True
                    _mark_last_transition_terminal(buffer, trajectory_ids[index])
                    buffer.fulfillment_rates.append(simulator.fulfillment_rate)
                    buffer.discard_count += simulator.discarded_count
                    buffer.completion_count += simulator.completed_count
                    buffer.fluid_solve_count += simulator.fluid_solve_count
                    buffer.fluid_solve_seconds += simulator.fluid_solve_seconds
                    buffer.fluid_cache_hit_count += simulator.fluid_cache_hit_count
                    buffer.fluid_fallback_count += simulator.fluid_fallback_count
                    buffer.episode_transition_counts.append(simulator.transition_count)
                    buffer.episode_seconds.append(time.perf_counter() - episode_start_times[index])
                    buffer.obs_build_seconds += simulator.obs_build_seconds
                    buffer.simulator_step_seconds += simulator.step_seconds
            continue

        obs_batch = [simulators[index].to_policy_obs() for index in active_indices]
        forward_start = time.perf_counter()
        with torch.no_grad():
            action_indexes, log_probs, values, env_actions = policy.act_batch(obs_batch, sample=True)
        forward_elapsed = time.perf_counter() - forward_start
        stats = getattr(policy, "last_forward_stats", {}) or {}
        batch_size = int(stats.get("batch_size", len(obs_batch)))
        policy_forward_count += 1
        policy_forward_seconds += float(stats.get("forward_seconds", forward_elapsed))
        total_requests += batch_size
        action_tokens_sum += float(stats.get("avg_action_tokens", 0.0)) * batch_size
        max_action_tokens = max(max_action_tokens, float(stats.get("max_action_tokens", 0.0)))
        padding_ratio_sum += float(stats.get("padding_ratio", 0.0)) * batch_size

        for local_index, env_index in enumerate(active_indices):
            simulator = simulators[env_index]
            reward = simulator.step(env_actions[local_index])
            done = simulator.done
            buffer.add(
                obs_batch[local_index],
                int(action_indexes[local_index].detach().cpu().item()),
                env_actions[local_index],
                float(log_probs[local_index].detach().cpu().item()),
                float(values[local_index].detach().cpu().item()),
                reward,
                done,
                trajectory_ids[env_index],
            )

    request_count = max(total_requests, 1)
    buffer.policy_forward_count = policy_forward_count
    buffer.policy_forward_seconds = policy_forward_seconds
    buffer.avg_action_tokens = action_tokens_sum / request_count
    buffer.max_action_tokens = max_action_tokens
    buffer.padding_ratio = padding_ratio_sum / request_count
    buffer.actor_batch_size_mean = total_requests / max(policy_forward_count, 1)
    return buffer
