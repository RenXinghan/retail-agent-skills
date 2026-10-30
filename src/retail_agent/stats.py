"""Per-task success rates and paired comparisons between runs on the same tasks."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from tau2.data_model.simulation import SimulationRun
from tau2.metrics.agent_metrics import is_successful


def task_success(sims: list[SimulationRun]) -> dict[str, tuple[int, int]]:
    """task id -> (successful trials, trials)."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for s in sims:
        out[s.task_id][0] += is_successful(s.reward_info.reward)
        out[s.task_id][1] += 1
    return {t: (k, n) for t, (k, n) in out.items()}


@dataclass
class PairedDelta:
    mean: float  # mean over tasks of (run success rate - baseline success rate)
    low: float  # 95% paired bootstrap interval over tasks
    high: float
    worse: int  # tasks where the run's success rate is lower
    better: int


def paired_delta(
    run: dict[str, tuple[int, int]],
    baseline: dict[str, tuple[int, int]],
    resamples: int = 10_000,
    seed: int = 0,
) -> PairedDelta:
    if set(run) != set(baseline):
        raise ValueError("Runs must cover the same tasks")
    tasks = sorted(run, key=lambda t: (0, int(t)) if t.isdigit() else (1, t))
    diffs = [run[t][0] / run[t][1] - baseline[t][0] / baseline[t][1] for t in tasks]
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(resamples))
    return PairedDelta(
        mean=sum(diffs) / n,
        low=boots[int(0.025 * resamples) - 1],
        high=boots[int(0.975 * resamples) - 1],
        worse=sum(d < 0 for d in diffs),
        better=sum(d > 0 for d in diffs),
    )
