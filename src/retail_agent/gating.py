"""Validation gating: keep only skills whose use did not hurt on held-out train tasks.

The agent runs once on the validation tasks with the full library. For every skill
we take the conversations in which it was loaded and compare their success with the
plain agent's success rate on the same tasks. This attributes outcomes to skills by
co-occurrence, so it is a cheap filter rather than a causal estimate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import SimulationRun
from tau2.metrics.agent_metrics import is_successful

from retail_agent.skills import SkillLibrary


@dataclass
class SkillEffect:
    name: str
    uses: int  # validation conversations that loaded the skill
    effect: float | None  # mean(success - plain-agent success rate of that task)
    kept: bool


def loaded_skills(sim: SimulationRun) -> set[str]:
    return {
        name
        for m in sim.messages or []
        if isinstance(m, AssistantMessage)
        for name in ((m.raw_data or {}).get("retail_agent", {}).get("skills_read") or [])
    }


def success_rates(sims: list[SimulationRun]) -> dict[str, float]:
    by_task: dict[str, list[bool]] = defaultdict(list)
    for s in sims:
        by_task[s.task_id].append(is_successful(s.reward_info.reward))
    return {t: sum(v) / len(v) for t, v in by_task.items()}


def gate(
    library: SkillLibrary,
    val_sims: list[SimulationRun],
    baseline_sims: list[SimulationRun],
    min_uses: int = 2,
) -> tuple[SkillLibrary, list[SkillEffect]]:
    """Keep skills loaded at least ``min_uses`` times whose effect is not negative."""
    baseline = success_rates(baseline_sims)
    deltas: dict[str, list[float]] = defaultdict(list)
    for sim in val_sims:
        if sim.task_id not in baseline:
            raise ValueError(f"No baseline runs for validation task {sim.task_id}")
        delta = is_successful(sim.reward_info.reward) - baseline[sim.task_id]
        for name in loaded_skills(sim):
            deltas[name].append(delta)
    effects = []
    for skill in library.skills:
        d = deltas.get(skill.name, [])
        effect = sum(d) / len(d) if d else None
        kept = len(d) >= min_uses and effect is not None and effect >= 0
        effects.append(SkillEffect(skill.name, len(d), effect, kept))
    kept = {e.name for e in effects if e.kept}
    return SkillLibrary([s for s in library.skills if s.name in kept]), effects
