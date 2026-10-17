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
    # names of checks that zero the reward when they fail (broken output schema, regressions)
    protected: Tuple[str, ...] = ()
    # workspace paths (fnmatch patterns, relative) the agent is meant to edit; every other file
    # present at reset must be unchanged at grading (outside output/)
    editable: Tuple[str, ...] = ()


def num_eq(x, y, tol: float = 0.01) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and abs(x - y) <= tol


def load_json_object(path: Path):
    """(object, ok): the parsed JSON object, or ({}, False) if missing or not an object."""
    try:
        got = __import__("json").loads(path.read_text())
        return (got, True) if isinstance(got, dict) else ({}, False)
    except Exception:
        return {}, False


# the output-format checks shared by the data families; failing them zeroes the reward
SCHEMA_CHECKS = ("output file exists and is a JSON object", "has exactly the required keys with the right types")
