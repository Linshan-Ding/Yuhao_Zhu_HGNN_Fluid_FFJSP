from html import escape
from numbers import Number


DEFAULT_METRICS = [
    "reward",
    "mean_return",
    "fulfillment_rate_mean",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "learning_rate",
    "steps_per_second",
    "transitions",
    "episode_count",
    "episode_transition_mean",
    "episode_transition_std",
    "rollout_seconds",
    "update_seconds",
    "episode_seconds_mean",
    "episode_seconds_std",
    "elapsed_seconds",
    "fluid_solve_count",
    "fluid_solve_seconds",
    "fluid_cache_hit_count",
    "fluid_fallback_count",
    "sparse_attention_enabled",
    "policy_forward_count",
    "policy_forward_seconds",
    "avg_action_tokens",
    "max_action_tokens",
    "padding_ratio",
    "actor_batch_size_mean",
    "policy_act_seconds",
    "simulator_step_seconds",
    "obs_build_seconds",
    "cpu_memory_mb",
    "gpu_memory_mb",
    "gpu_temperature",
    "completion_count",
    "discard_count",
]


class TrainingVisualizer:
    def __init__(self, train_paras: dict):
        self.enabled = bool(train_paras.get("viz", False))
        self.env = train_paras.get("viz_name", "HGNN_PPO")
        self.server = train_paras.get("visdom_server", "http://localhost")
        self.port = int(train_paras.get("visdom_port", 8097))
        self.update_interval = max(int(train_paras.get("viz_update_interval", 1)), 1)
        self.metrics = self._load_metrics(train_paras.get("viz_metrics"))
        self.viz = None
        self.windows = set()
        self.warned = False
        if self.enabled:
            self._connect()

    def _load_metrics(self, configured_metrics):
        if not configured_metrics:
            return DEFAULT_METRICS
        return [str(metric) for metric in configured_metrics]

    def _connect(self):
        try:
            from visdom import Visdom

            self.viz = Visdom(server=self.server, port=self.port, env=self.env)
            if not self.viz.check_connection(timeout_seconds=1):
                self.viz = None
                self._warn_unavailable()
        except Exception as exc:
            self.viz = None
            self._warn_unavailable(exc)

    def _warn_unavailable(self, exc: Exception | None = None):
        if self.warned:
            return
        detail = f" ({exc})" if exc else ""
        print(
            f"[Visdom] unavailable at {self.server}:{self.port}{detail}; "
            "training will continue with CSV/console logging."
        )
        self.warned = True

    def log(self, step: int, metrics: dict):
        if not self.enabled or step % self.update_interval != 0:
            return
        if self.viz is None:
            return
        for metric_name in self.metrics:
            self._log_metric(step, metric_name, metrics)
        self._log_instance(step, metrics)

    def _log_metric(self, step: int, metric_name: str, metrics: dict):
        value = metrics.get(metric_name)
        if not isinstance(value, Number):
            return

        win = metric_name
        opts = {
            "title": metric_name,
            "xlabel": "iteration",
            "ylabel": "value",
        }
        update = "append" if win in self.windows else None
        self.viz.line([float(value)], [step], win=win, update=update, opts=opts)
        self.windows.add(win)

    def _log_instance(self, step: int, metrics: dict):
        instance_text = (
            f"<b>Iteration:</b> {step}<br>"
            f"<b>Instance ID:</b> {metrics.get('instance_id', '')}<br>"
            f"<b>Instance:</b> {escape(str(metrics.get('instance_name', '')))}<br>"
            f"<b>Selection:</b> {escape(str(metrics.get('instance_selection', '')))}<br>"
            f"<b>Fulfillment Mean:</b> {float(metrics.get('fulfillment_rate_mean', 0.0)):.6f}"
        )
        self.viz.text(instance_text, win="instance", opts={"title": "instance"})
        self.windows.add("instance")

    def close(self):
        self.viz = None
