"""Skill library: storage, retrieval and prompt rendering."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from rank_bm25 import BM25Plus


class Skill(BaseModel):
    """A reusable procedure distilled from agent trajectories."""

    name: str = Field(description="Short snake_case identifier, unique in the library.")
    description: str = Field(description="One line: what the skill helps with.")
    when_to_use: list[str] = Field(default_factory=list)
    steps: list[str] = Field(min_length=1)
    pitfalls: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(
        default_factory=list, description="Task ids whose trajectories produced this skill."
    )
    version: int = 1

    def render(self) -> str:
        lines = [f"## {self.name}", self.description]
        if self.when_to_use:
            lines.append("When to use:")
            lines += [f"- {w}" for w in self.when_to_use]
        lines.append("Steps:")
        lines += [f"{i}. {s}" for i, s in enumerate(self.steps, 1)]
        if self.pitfalls:
            lines.append("Pitfalls:")
            lines += [f"- {p}" for p in self.pitfalls]
        return "\n".join(lines)

    def search_text(self) -> str:
        return " ".join([self.name.replace("_", " "), self.description, *self.when_to_use])


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Bm25Index:
    """BM25+ ranking that only returns documents sharing a token with the query.

    BM25+ keeps IDF positive, which matters for libraries of a dozen skills where
    plain BM25 gives common words negative weight.
    """

    def __init__(self, texts: list[str]):
        self._docs = [set(tokenize(t)) for t in texts]
        self._bm25 = BM25Plus([tokenize(t) for t in texts]) if texts else None

    def search(self, query: str, k: int) -> list[int]:
        if self._bm25 is None or k <= 0:
            return []
        q = tokenize(query)
        scores = self._bm25.get_scores(q)
        hits = [i for i, doc in enumerate(self._docs) if doc.intersection(q)]
        return sorted(hits, key=lambda i: scores[i], reverse=True)[:k]


class SkillLibrary:
    def __init__(self, skills: list[Skill]):
        names = [s.name for s in skills]
        if len(names) != len(set(names)):
            raise ValueError("Skill names must be unique")
        self.skills = skills
        self._by_name = {s.name: s for s in skills}
        self._index = Bm25Index([s.search_text() for s in skills])

    @classmethod
    def load(cls, path: str | Path) -> SkillLibrary:
        data = yaml.safe_load(Path(path).read_text()) or []
        return cls([Skill.model_validate(d) for d in data])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = [s.model_dump() for s in self.skills]
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    def __len__(self) -> int:
        return len(self.skills)

    def get(self, name: str) -> Skill | None:
        return self._by_name.get(name)

    def search(self, query: str, k: int) -> list[Skill]:
        return [self.skills[i] for i in self._index.search(query, k)]

    def render_index(self) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in self.skills)
