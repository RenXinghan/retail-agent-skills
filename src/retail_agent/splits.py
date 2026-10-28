"""Fixed split of the tau2 retail train tasks into a mining part and a validation part."""

from __future__ import annotations

from functools import cache

from tau2.runner import get_tasks

PARTS = ("mine", "val")


@cache
def train_part(part: str) -> list[str]:
    """Every third train task (by id) is held out for validation; the rest are for mining."""
    if part not in PARTS:
        raise ValueError(f"Unknown part {part!r}")
    ids = sorted((t.id for t in get_tasks("retail", task_split_name="train")), key=int)
    val = ids[::3]
    return val if part == "val" else [i for i in ids if i not in set(val)]
