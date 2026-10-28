"""Compact trajectories built from tau2 results, plus a BM25 store over them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import Results, SimulationRun
from tau2.metrics.agent_metrics import is_successful

from retail_agent.skills import Bm25Index

TOOL_RESULT_CHARS = 400


@dataclass
class Trajectory:
    task_id: str
    trial: int
    reward: float
    user_text: str
    transcript: str

    @property
    def success(self) -> bool:
        return is_successful(self.reward)

    def render(self) -> str:
        return f"### Example (task {self.task_id})\n{self.transcript}"


def render_transcript(sim: SimulationRun, tool_result_chars: int = TOOL_RESULT_CHARS) -> str:
    lines = []
    for m in sim.messages or []:
        if isinstance(m, UserMessage) and m.content:
            lines.append(f"User: {m.content}")
        elif isinstance(m, AssistantMessage):
            if m.content:
                lines.append(f"Agent: {m.content}")
            for c in m.tool_calls or []:
                lines.append(f"Agent calls {c.name}({json.dumps(c.arguments, ensure_ascii=False)})")
        elif isinstance(m, ToolMessage):
            content = (m.content or "").replace("\n", " ")
            if len(content) > tool_result_chars:
                content = content[:tool_result_chars] + " ..."
            lines.append(f"{'Tool error' if m.error else 'Tool'}: {content}")
    return "\n".join(lines)


def from_simulation(sim: SimulationRun) -> Trajectory:
    user_text = "\n".join(
        m.content for m in sim.messages or [] if isinstance(m, UserMessage) and m.content
    )
    return Trajectory(
        task_id=sim.task_id,
        trial=sim.trial or 0,
        reward=sim.reward_info.reward if sim.reward_info else 0.0,
        user_text=user_text,
        transcript=render_transcript(sim),
    )


def load_trajectories(paths: list[str | Path]) -> list[Trajectory]:
    return [from_simulation(sim) for p in paths for sim in Results.load(Path(p)).simulations]


class TrajectoryStore:
    """Successful trajectories, one per task, retrievable by the user's words."""

    def __init__(self, trajectories: list[Trajectory]):
        seen: dict[str, Trajectory] = {}
        for t in trajectories:
            if t.success and t.task_id not in seen:
                seen[t.task_id] = t
        self.items = list(seen.values())
        self._index = Bm25Index([t.user_text for t in self.items])

    def __len__(self) -> int:
        return len(self.items)

    @property
    def task_ids(self) -> set[str]:
        return {t.task_id for t in self.items}

    def search(self, query: str, k: int) -> list[Trajectory]:
        return [self.items[i] for i in self._index.search(query, k)]
