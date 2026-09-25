"""Public scheduling actions. Observations live in environment.public."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Dispatch:
    order: int
    machine: int

@dataclass(frozen=True)
class Wait:
    pass
