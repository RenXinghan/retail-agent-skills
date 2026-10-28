"""A tau2 half-duplex agent whose per-turn control flow is a LangGraph state graph.

Each call to ``generate_next_message`` runs the graph once:

    ingest -> [retrieve | route -> read_skills] -> act <-> read_skills -> finalize

``act`` is one LLM call. In ``skills_index`` mode the model only sees a one-line
index of the skill library and loads full procedures with an internal
``read_skills`` tool. Left optional, the model never used that tool in our pilot,
so on every customer turn ``route`` forces one ``read_skills`` call in which the
model picks the matching skills (possibly none); it may load more later. Those
calls are answered inside the graph and never reach the environment, so they
cannot affect the evaluated trajectory.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import AGENT_INSTRUCTION
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import generate

from retail_agent.skills import SkillLibrary
from retail_agent.trajectories import TrajectoryStore


class Mode(StrEnum):
    VANILLA = "vanilla"
    TRAJECTORIES = "trajectories"  # top-k raw train trajectories as examples
    SKILLS_INJECT = "skills_inject"  # top-k full skills in the system prompt
    SKILLS_INDEX = "skills_index"  # skill index + model-selected read_skills


READ_SKILLS = "read_skills"
MAX_SKILL_READS_PER_STEP = 3

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

SKILLS_INJECT_BLOCK = """
<skills>
Procedures learned from earlier conversations. Use them when they apply; the policy
above always takes precedence.

{skills}
</skills>
""".strip()

SKILLS_INDEX_BLOCK = """
<skill_index>
Procedures learned from earlier conversations, listed by name. They hold steps and
pitfalls that the policy alone does not spell out. Use `read_skills` to load the ones that
match the user's request. Loading skills is internal: it is invisible to the user and does
not use up your turn. The policy above always takes precedence.

{index}
</skill_index>
""".strip()

TRAJECTORIES_BLOCK = """
<examples>
Transcripts of earlier conversations that were resolved correctly. They show how
the policy was applied; the details of the current user will differ.

{examples}
</examples>
""".strip()


def read_skills(names: list[str]) -> str:
    """Load the full procedures of skills from the skill index.

    Args:
        names: Exact skill names from the skill index that match the user's request;
            an empty list if none match.

    Returns:
        The steps and pitfalls of each skill.
    """
    raise NotImplementedError("Answered inside the agent graph")


class SkillAgentState(BaseModel):
    """Conversation state carried between turns by the tau2 orchestrator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[APICompatibleMessage]
    context: str = ""
    skills_read: list[str] = []


class TurnState(TypedDict, total=False):
    incoming: ValidAgentInputMessage
    messages: list[APICompatibleMessage]
    context: str
    retrieved: list[str]
    response: AssistantMessage
    calls: list[AssistantMessage]
    reads: int  # read_skills calls this turn
    step_reads: list[str]
    skills_read: list[str]


class SkillAgent(HalfDuplexAgent[SkillAgentState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: dict | None = None,
        mode: Mode = Mode.VANILLA,
        library: SkillLibrary | None = None,
        store: TrajectoryStore | None = None,
        top_k: int = 3,
        generate_fn: Callable[..., AssistantMessage] = generate,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        if mode in (Mode.SKILLS_INJECT, Mode.SKILLS_INDEX) and not library:
            raise ValueError(f"Mode {mode} needs a non-empty skill library")
        if mode == Mode.TRAJECTORIES and not store:
            raise ValueError("Mode trajectories needs a trajectory store")
        self.llm = llm
        self.llm_args = llm_args or {}
        self.mode = Mode(mode)
        self.library = library
        self.store = store
        self.top_k = top_k
        self._generate = generate_fn
        self._read_skills_tool = as_tool(read_skills)
        self._graph = self._build_graph()

    # ---- tau2 interface ----

    def get_init_state(self, message_history: list[Message] | None = None) -> SkillAgentState:
        return SkillAgentState(messages=list(message_history or []))

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: SkillAgentState
    ) -> tuple[AssistantMessage, SkillAgentState]:
        out = self._graph.invoke(
            {
                "incoming": message,
                "messages": state.messages,
                "context": state.context,
                "skills_read": state.skills_read,
            }
        )
        new_state = SkillAgentState(
            messages=out["messages"], context=out["context"], skills_read=out["skills_read"]
        )
        return out["response"], new_state

    # ---- graph ----

    def _build_graph(self):
        g = StateGraph(TurnState)
        g.add_node("ingest", self._ingest)
        g.add_node("retrieve", self._retrieve)
        g.add_node("route", self._route)
        g.add_node("act", self._act)
        g.add_node("read_skills", self._read_skills)
        g.add_node("finalize", self._finalize)
        g.add_edge(START, "ingest")
        g.add_conditional_edges("ingest", self._after_ingest, ["retrieve", "route", "act"])
        g.add_edge("retrieve", "act")
        g.add_conditional_edges(
            "route", lambda s: "read_skills" if self._internal(s) else "act", ["read_skills", "act"]
        )
        g.add_conditional_edges(
            "act",
            lambda s: "read_skills" if self._internal(s) else "finalize",
            ["read_skills", "finalize"],
        )
        g.add_edge("read_skills", "act")
        g.add_edge("finalize", END)
        return g.compile()

    def _ingest(self, s: TurnState) -> TurnState:
        incoming = s["incoming"]
        new = incoming.tool_messages if isinstance(incoming, MultiToolMessage) else [incoming]
        return {"messages": s["messages"] + list(new), "calls": [], "reads": 0, "step_reads": []}

    def _after_ingest(self, s: TurnState) -> str:
        if not isinstance(s["incoming"], UserMessage):
            return "act"
        if self.mode in (Mode.SKILLS_INJECT, Mode.TRAJECTORIES):
            return "retrieve"
        return "route" if self.mode == Mode.SKILLS_INDEX else "act"

    def _retrieve(self, s: TurnState) -> TurnState:
        query = "\n".join(
            m.content for m in s["messages"] if isinstance(m, UserMessage) and m.content
        )
        if self.mode == Mode.SKILLS_INJECT:
            hits = self.library.search(query, self.top_k)
            body = "\n\n".join(h.render() for h in hits)
            context = SKILLS_INJECT_BLOCK.format(skills=body) if hits else ""
            return {"context": context, "retrieved": [h.name for h in hits]}
        hits = self.store.search(query, self.top_k)
        body = "\n\n".join(h.render() for h in hits)
        context = TRAJECTORIES_BLOCK.format(examples=body) if hits else ""
        return {"context": context, "retrieved": [h.task_id for h in hits]}

    def _system_messages(self, context: str) -> list[SystemMessage]:
        prompt = SYSTEM_PROMPT.format(
            agent_instruction=AGENT_INSTRUCTION, domain_policy=self.domain_policy
        )
        if self.mode == Mode.SKILLS_INDEX:
            context = SKILLS_INDEX_BLOCK.format(index=self.library.render_index())
        if context:
            prompt = f"{prompt}\n{context}"
        return [SystemMessage(role="system", content=prompt)]

    def _tools(self, reads: int) -> list[Tool]:
        if self.mode == Mode.SKILLS_INDEX and reads < MAX_SKILL_READS_PER_STEP:
            return [*self.tools, self._read_skills_tool]
        return self.tools

    def _route(self, s: TurnState) -> TurnState:
        response = self._generate(
            model=self.llm,
            tools=self._tools(0),
            tool_choice={"type": "function", "function": {"name": READ_SKILLS}},
            messages=self._system_messages(s["context"]) + s["messages"],
            call_name="skill_agent_route",
            **self.llm_args,
        )
        return {"response": response, "calls": s["calls"] + [response]}

    def _act(self, s: TurnState) -> TurnState:
        response = self._generate(
            model=self.llm,
            tools=self._tools(s["reads"]),
            messages=self._system_messages(s["context"]) + s["messages"],
            call_name="skill_agent_act",
            **self.llm_args,
        )
        return {"response": response, "calls": s["calls"] + [response]}

    def _internal(self, s: TurnState) -> bool:
        return any(c.name == READ_SKILLS for c in s["response"].tool_calls or [])

    def _read_skills(self, s: TurnState) -> TurnState:
        # Keep only the internal calls: environment calls made in the same message are
        # dropped so history stays consistent; the model re-issues them on the next act.
        response = s["response"]
        internal = [c for c in response.tool_calls if c.name == READ_SKILLS]
        call_msg = AssistantMessage(role="assistant", content=response.content, tool_calls=internal)
        results, read = [], []
        for c in internal:
            names = c.arguments.get("names") or []
            names = [names] if isinstance(names, str) else [str(n) for n in names]
            parts, unknown = [], []
            for name in names:
                if name in s["skills_read"] or name in read:
                    parts.append(f"## {name}\n(already loaded earlier in this conversation)")
                elif skill := self.library.get(name):
                    read.append(name)
                    parts.append(skill.render())
                else:
                    unknown.append(name)
            if unknown:
                parts.append(
                    f"Unknown skills {unknown}. Available: "
                    + ", ".join(sk.name for sk in self.library.skills)
                )
            results.append(
                ToolMessage(
                    id=c.id,
                    role="tool",
                    content="\n\n".join(parts) or "No skills loaded.",
                    requestor="assistant",
                    error=bool(unknown),
                )
            )
        return {
            "messages": s["messages"] + [call_msg, *results],
            "reads": s["reads"] + len(internal),
            "step_reads": s["step_reads"] + read,
            "skills_read": s["skills_read"] + read,
        }

    def _finalize(self, s: TurnState) -> TurnState:
        response = s["response"]
        calls = s["calls"]
        costs = [c.cost for c in calls]
        response.cost = None if any(c is None for c in costs) else sum(costs)
        usages = [c.usage for c in calls if c.usage]
        if usages:
            response.usage = {
                k: sum(u.get(k) or 0 for u in usages)
                for k in ("prompt_tokens", "completion_tokens")
            }
        # Provider usage (incl. cache hits) of every call, before raw_data is replaced.
        raw_usage = [(c.raw_data or {}).get("usage") or {} for c in calls]
        cache_hits = sum(u.get("prompt_cache_hit_tokens") or 0 for u in raw_usage)
        response.raw_data = {
            **(response.raw_data or {}),
            "retail_agent": {
                "mode": self.mode.value,
                "llm_calls": len(calls),
                "retrieved": s.get("retrieved"),
                "skills_read": s["step_reads"],
                "prompt_cache_hit_tokens": cache_hits,
            },
        }
        return {"messages": s["messages"] + [response], "response": response}


def create_skill_agent(tools: list[Tool], domain_policy: str, **kwargs: Any) -> SkillAgent:
    """tau2 agent factory; bind mode/library/store/top_k with functools.partial."""
    return SkillAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs["llm"],
        llm_args=kwargs.get("llm_args"),
        mode=kwargs.get("mode", Mode.VANILLA),
        library=kwargs.get("library"),
        store=kwargs.get("store"),
        top_k=kwargs.get("top_k", 3),
    )
