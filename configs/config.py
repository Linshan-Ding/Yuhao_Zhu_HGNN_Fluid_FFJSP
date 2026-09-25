"""配置加载：合并多份 YAML -> 结构化对象。代码只从这里读参数，不出现魔法数字。"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"



def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class Config:
    """点号访问的配置树；`raw` 保留原始 dict 以便快照落盘。"""

    raw: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, item: str) -> Any:
        raw = object.__getattribute__(self, "raw")
        if item not in raw:
            raise AttributeError(f"config has no key '{item}'")
        value = raw[item]
        return Config(value) if isinstance(value, dict) else value

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)

    def snapshot(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.raw, handle, allow_unicode=True, sort_keys=False)

