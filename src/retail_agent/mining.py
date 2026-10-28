"""Distill a skill library from the agent's own train-split trajectories.

The miner only sees what a deployed agent could see: its transcripts and a
pass/fail outcome per conversation. It never sees the benchmark's gold actions.

1. reflect:     per task, compare the agent's successful and failed attempts (or
                study a lone success/failure) and propose candidate skills.
2. consolidate: merge near-duplicate candidates into a library of at most
                ``max_skills`` skills, keeping provenance.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger
from pydantic import ValidationError
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.utils.llm_utils import generate

from retail_agent.skills import Skill, SkillLibrary
from retail_agent.trajectories import Trajectory

SKILL_FORMAT = """
Each skill is a JSON object:
{
  "name": "snake_case_name",
  "description": "one line: which kind of request this procedure handles",
  "when_to_use": ["observable cue in the conversation", ...],
  "steps": ["imperative step, naming the tools to call", ...],
  "pitfalls": ["a concrete mistake to avoid", ...]
}
Skills must generalize: never include user names, ids, order numbers, prices or
other details that are specific to one conversation.
""".strip()

REFLECT_SYSTEM = f"""
You improve a customer-service agent by turning its past conversations into
reusable procedures ("skills"). You receive the domain policy and one or more
attempts at the same customer request, each marked SUCCESS or FAILURE by an
automatic grader that checks the final database state and required replies.

When both outcomes exist, find what the successful attempt did differently and
write that as steps or pitfalls. For a lone failure, infer the most likely
mistake from the policy and the transcript. Propose at most 2 skills, or none if
the conversation teaches nothing beyond the policy text.

{SKILL_FORMAT}

Answer with JSON only: {{"skills": [ ... ]}}
""".strip()

CONSOLIDATE_SYSTEM = f"""
You maintain a skill library for a customer-service agent. Merge the candidate
skills below into at most <MAX_SKILLS> skills: combine duplicates and
near-duplicates, keep the most specific and actionable steps and pitfalls, drop
skills that only restate the policy, and keep each skill focused on one kind of
request. For every output skill, list in "provenance" the union of the
"provenance" task ids of the candidates it was built from.

{SKILL_FORMAT}
Also include "provenance": ["task id", ...] in each skill.

Answer with JSON only: {{"skills": [ ... ]}}
""".strip()


def _parse_json(text: str) -> Any:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    return json.loads(text)


def _ask_json(generate_fn, model: str, llm_args: dict, system: str, user: str, retries: int = 1):
    messages = [
        SystemMessage(role="system", content=system),
        UserMessage(role="user", content=user),
    ]
    for attempt in range(retries + 1):
        reply = generate_fn(model=model, messages=messages, call_name="skill_mining", **llm_args)
        try:
            return _parse_json(reply.content or "")
        except json.JSONDecodeError:
            logger.warning(f"Unparseable miner reply (attempt {attempt + 1})")
    return None


def _to_skills(raw: Any, provenance: list[str] | None = None) -> list[Skill]:
    skills = []
    for item in (raw or {}).get("skills", []):
        if provenance is not None:
            item["provenance"] = provenance
        try:
            skills.append(Skill.model_validate(item))
        except ValidationError as e:
            logger.warning(f"Dropping malformed skill: {e.errors()[0]['msg']}")
    return skills


def _attempts_prompt(policy: str, attempts: list[Trajectory]) -> str:
    parts = [f"<policy>\n{policy}\n</policy>"]
    for i, t in enumerate(attempts, 1):
        outcome = "SUCCESS" if t.success else "FAILURE"
        parts.append(f"<attempt {i}: {outcome}>\n{t.transcript}\n</attempt {i}>")
    return "\n\n".join(parts)


def pick_attempts(trajectories: list[Trajectory]) -> list[Trajectory]:
    """One success and one failure when both exist, else a single attempt."""
    successes = [t for t in trajectories if t.success]
    failures = [t for t in trajectories if not t.success]
    if successes and failures:
        return [successes[0], failures[0]]
    return [(successes or failures)[0]]


def reflect(
    policy: str,
    trajectories: list[Trajectory],
    model: str,
    llm_args: dict,
    concurrency: int = 4,
    generate_fn: Callable = generate,
) -> list[Skill]:
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_task[t.task_id].append(t)

    def one(task_id: str) -> list[Skill]:
        attempts = pick_attempts(by_task[task_id])
        raw = _ask_json(
            generate_fn, model, llm_args, REFLECT_SYSTEM, _attempts_prompt(policy, attempts)
        )
        return _to_skills(raw, provenance=[task_id])

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, sorted(by_task)))
    return [s for skills in results for s in skills]


def consolidate(
    candidates: list[Skill],
    model: str,
    llm_args: dict,
    max_skills: int,
    generate_fn: Callable = generate,
) -> SkillLibrary:
    payload = json.dumps(
        [c.model_dump(exclude={"version"}) for c in candidates], ensure_ascii=False, indent=1
    )
    raw = _ask_json(
        generate_fn,
        model,
        llm_args,
        CONSOLIDATE_SYSTEM.replace("<MAX_SKILLS>", str(max_skills)),
        f"Candidate skills:\n{payload}",
    )
    merged: dict[str, Skill] = {}
    for s in _to_skills(raw)[:max_skills]:
        merged.setdefault(s.name, s)
    return SkillLibrary(list(merged.values()))


def mine(
    policy: str,
    trajectories: list[Trajectory],
    model: str,
    llm_args: dict | None = None,
    max_skills: int = 12,
    concurrency: int = 4,
    generate_fn: Callable = generate,
) -> tuple[SkillLibrary, list[Skill]]:
    llm_args = llm_args or {}
    candidates = reflect(policy, trajectories, model, llm_args, concurrency, generate_fn)
    logger.info(f"{len(candidates)} candidate skills from {len(trajectories)} trajectories")
    if not candidates:
        return SkillLibrary([]), []
    return consolidate(candidates, model, llm_args, max_skills, generate_fn), candidates
