"""Types shared by all task families."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple

Check = Tuple[str, bool]


@dataclass
class TaskInstance:
    instruction: str
    verify: Callable[[Path], List[Check]]
    meta: dict = field(default_factory=dict)  # ground truth, for tests and debugging only


def num_eq(x, y, tol: float = 0.01) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and abs(x - y) <= tol


def load_json_object(path: Path):
    """(object, ok): the parsed JSON object, or ({}, False) if missing or not an object."""
    try:
        got = __import__("json").loads(path.read_text())
        return (got, True) if isinstance(got, dict) else ({}, False)
    except Exception:
        return {}, False
