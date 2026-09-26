"""Dot-access view of the simulator configuration built by configs.experiment.environment_config."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Config:
    """点号访问的配置树；`raw` 保留原始 dict 以便随模拟器状态落盘。缺失键直接报错，不提供默认值。"""

    raw: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, item: str) -> Any:
        raw = object.__getattribute__(self, "raw")
        if item not in raw:
            raise AttributeError(f"config has no key '{item}'")
        value = raw[item]
        return Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)
