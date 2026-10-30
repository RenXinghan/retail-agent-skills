"""Offline tests: every LLM call is replaced by a scripted fake."""

import json
from types import SimpleNamespace

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.runner import build_environment

from retail_agent.agent import Mode, SkillAgent
from retail_agent.cli import leaked_tasks
from retail_agent.mining import mine, pick_attempts
from retail_agent.skills import Skill, SkillLibrary
from retail_agent.trajectories import Trajectory, TrajectoryStore

EXCHANGE = Skill(
    name="exchange_delivered_items",
    description="Exchange items of a delivered order for other variants of the same product",
    when_to_use=["user wants to exchange or swap an item they received"],
    steps=["Authenticate the user", "Collect every item to exchange before calling the tool"],
    pitfalls=["The exchange tool can only be called once per order"],
    provenance=["7"],
)
CANCEL = Skill(
    name="cancel_pending_order",
    description="Cancel an order that has not shipped yet",
    when_to_use=["user wants to cancel an order"],
    steps=["Check the order status is pending", "Ask for the cancellation reason"],
    provenance=["9"],
)


class FakeLLM:
    """Returns scripted AssistantMessages in order and records every call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, model, messages, tools=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tools": [t.name for t in tools or []],
                "tool_choice": kwargs.get("tool_choice"),
            }
        )
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            reply = AssistantMessage(role="assistant", content=reply)
        reply.cost = 0.001
        reply.usage = {"prompt_tokens": 100, "completion_tokens": 10}
        return reply


def tool_call(tool, **arguments):
    return AssistantMessage(
        role="assistant", tool_calls=[ToolCall(id=f"call_{tool}", name=tool, arguments=arguments)]
    )


@pytest.fixture(scope="module")
def env():
    return build_environment("retail")


def make_agent(env, mode, llm, **kwargs):
    return SkillAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm="fake",
        mode=mode,
        generate_fn=llm,
        **kwargs,
    )


def user(text):
    return UserMessage(role="user", content=text)


def test_library_search_and_roundtrip(tmp_path):
    lib = SkillLibrary([EXCHANGE, CANCEL])
    assert lib.search("I want to exchange my keyboard", 1)[0].name == EXCHANGE.name
    assert lib.search("zzz", 2) == []
    lib.save(tmp_path / "s.yaml")
    assert SkillLibrary.load(tmp_path / "s.yaml").get(CANCEL.name) == CANCEL
    with pytest.raises(ValueError):
        SkillLibrary([CANCEL, CANCEL])


def test_vanilla_is_one_plain_llm_call(env):
    llm = FakeLLM(["Hello, how can I help?"])
    agent = make_agent(env, Mode.VANILLA, llm)
    msg, state = agent.generate_next_message(user("hi"), agent.get_init_state())
    assert msg.content == "Hello, how can I help?"
    assert len(llm.calls) == 1
    assert "read_skills" not in llm.calls[0]["tools"]
    assert "<skill" not in llm.calls[0]["messages"][0].content
    assert [m.role for m in state.messages] == ["user", "assistant"]


def test_skills_inject_puts_relevant_skill_in_prompt(env):
    llm = FakeLLM(["Sure, let me look up your account."])
    agent = make_agent(
        env, Mode.SKILLS_INJECT, llm, library=SkillLibrary([EXCHANGE, CANCEL]), top_k=1
    )
    msg, _ = agent.generate_next_message(
        user("I need to exchange a keyboard"), agent.get_init_state()
    )
    system = llm.calls[0]["messages"][0].content
    assert "## exchange_delivered_items" in system
    assert "## cancel_pending_order" not in system
    assert msg.raw_data["retail_agent"]["retrieved"] == ["exchange_delivered_items"]


def test_skills_index_routes_then_acts(env):
    llm = FakeLLM(
        [
            tool_call("read_skills", names=["exchange_delivered_items"]),
            tool_call("find_user_id_by_email", email="a@b.com"),
        ]
    )
    agent = make_agent(env, Mode.SKILLS_INDEX, llm, library=SkillLibrary([EXCHANGE, CANCEL]))
    msg, state = agent.generate_next_message(
        user("exchange please, a@b.com"), agent.get_init_state()
    )

    # The first call is forced to choose skills; only the environment call leaves the agent.
    assert llm.calls[0]["tool_choice"]["function"]["name"] == "read_skills"
    assert llm.calls[1]["tool_choice"] is None
    assert [c.name for c in msg.tool_calls] == ["find_user_id_by_email"]
    # The skill body was visible to the second LLM call, the index to both.
    second = llm.calls[1]["messages"]
    assert isinstance(second[-1], ToolMessage) and "only be called once" in second[-1].content
    assert "- cancel_pending_order:" in second[0].content
    # Cost and tokens cover both LLM calls.
    assert msg.cost == pytest.approx(0.002)
    assert msg.usage == {"prompt_tokens": 200, "completion_tokens": 20}
    assert msg.raw_data["retail_agent"]["skills_read"] == ["exchange_delivered_items"]
    assert state.skills_read == ["exchange_delivered_items"]
    assert [m.role for m in state.messages] == ["user", "assistant", "tool", "assistant"]


def test_skills_index_routes_only_on_user_turns_and_skips_loaded(env):
    lib = SkillLibrary([EXCHANGE])
    llm = FakeLLM(
        [
            tool_call("read_skills", names=["exchange_delivered_items"]),
            tool_call("find_user_id_by_email", email="a@b.com"),
            "Found you. Which items?",  # after the tool result: no routing
            tool_call("read_skills", names=["exchange_delivered_items"]),  # next user turn
            "Got it.",
        ]
    )
    agent = make_agent(env, Mode.SKILLS_INDEX, llm, library=lib)
    _, state = agent.generate_next_message(user("exchange, a@b.com"), agent.get_init_state())
    result = ToolMessage(id="call_find_user_id_by_email", role="tool", content="u1")
    _, state = agent.generate_next_message(result, state)
    msg, state = agent.generate_next_message(user("the keyboard"), state)
    assert [c["tool_choice"] is not None for c in llm.calls] == [True, False, False, True, False]
    assert "already loaded" in state.messages[-2].content
    assert state.skills_read == ["exchange_delivered_items"]
    assert msg.content == "Got it."


def test_skills_index_route_may_choose_nothing(env):
    llm = FakeLLM([tool_call("read_skills", names=[]), "Could you share your email?"])
    agent = make_agent(env, Mode.SKILLS_INDEX, llm, library=SkillLibrary([EXCHANGE]))
    msg, state = agent.generate_next_message(user("hello"), agent.get_init_state())
    assert msg.content == "Could you share your email?"
    assert state.messages[2].content == "No skills loaded."
    assert msg.raw_data["retail_agent"]["skills_read"] == []


def test_skills_index_caps_reads_and_handles_unknown_names(env):
    reads = [tool_call("read_skills", names=["nope"]) for _ in range(3)]
    llm = FakeLLM([*reads, "Could you share your email?"])
    agent = make_agent(env, Mode.SKILLS_INDEX, llm, library=SkillLibrary([EXCHANGE]))
    msg, state = agent.generate_next_message(user("hello"), agent.get_init_state())
    assert msg.content == "Could you share your email?"
    assert "read_skills" in llm.calls[-2]["tools"]
    assert "read_skills" not in llm.calls[-1]["tools"]
    errors = [m for m in state.messages if isinstance(m, ToolMessage)]
    assert len(errors) == 3 and all(m.error for m in errors)


def traj(task_id, reward, user_text="exchange keyboard", trial=0):
    return Trajectory(task_id, trial, reward, user_text, f"User: {user_text}\nAgent: ok")


def test_trajectory_store_keeps_one_success_per_task():
    store = TrajectoryStore(
        [traj("1", 0.0), traj("1", 1.0), traj("1", 1.0, trial=1), traj("2", 0.0)]
    )
    assert [t.task_id for t in store.items] == ["1"]
    assert store.search("keyboard", 3)[0].task_id == "1"


def test_leak_guard():
    lib = SkillLibrary([EXCHANGE])
    store = TrajectoryStore([traj("3", 1.0)])
    assert leaked_tasks({"3", "7", "8"}, lib, store) == ["3", "7"]
    assert leaked_tasks({"8"}, lib, store) == []


def test_pick_attempts_prefers_contrast():
    picked = pick_attempts([traj("1", 0.0), traj("1", 1.0, trial=1), traj("1", 0.0, trial=2)])
    assert sorted(t.success for t in picked) == [False, True]


def test_mine_reflects_per_task_then_consolidates():
    candidate = EXCHANGE.model_dump(exclude={"provenance", "version"})
    merged = {**candidate, "provenance": ["1", "2"]}
    llm = FakeLLM(
        [
            "```json\n" + json.dumps({"skills": [candidate]}) + "\n```",
            json.dumps({"skills": [candidate, {"name": "bad"}]}),  # malformed one is dropped
            json.dumps({"skills": [merged]}),
        ]
    )
    library, candidates = mine(
        "policy", [traj("1", 1.0), traj("2", 0.0)], model="fake", concurrency=1, generate_fn=llm
    )
    assert [c.provenance for c in candidates] == [["1"], ["2"]]
    assert len(library) == 1 and library.skills[0].provenance == ["1", "2"]
    assert len(llm.calls) == 3


def test_failures_only_mining_skips_clean_tasks():
    candidate = EXCHANGE.model_dump(exclude={"provenance", "version"})
    llm = FakeLLM([json.dumps({"skills": [candidate]}), json.dumps({"skills": [candidate]})])
    _, candidates = mine(
        "policy",
        [traj("1", 1.0), traj("1", 1.0, trial=1), traj("2", 1.0), traj("2", 0.0, trial=1)],
        model="fake",
        concurrency=1,
        generate_fn=llm,
        failures_only=True,
    )
    assert [c.provenance for c in candidates] == [["2"]]
    assert "learning from its mistakes" in llm.calls[0]["messages"][0].content


def test_train_parts_are_disjoint_and_cover_train():
    from retail_agent.splits import train_part

    mine_ids, val_ids = set(train_part("mine")), set(train_part("val"))
    assert not mine_ids & val_ids
    assert len(mine_ids | val_ids) == 74 and len(val_ids) == 25


def sim(task_id, reward, skills=()):
    from tau2.data_model.simulation import RewardInfo

    msg = AssistantMessage(
        role="assistant", content="ok", raw_data={"retail_agent": {"skills_read": list(skills)}}
    )
    return SimpleNamespace(task_id=task_id, reward_info=RewardInfo(reward=reward), messages=[msg])


def test_gate_drops_skills_used_in_regressions():
    from retail_agent.gating import gate

    lib = SkillLibrary([EXCHANGE, CANCEL, Skill(name="unused", description="x", steps=["y"])])
    baseline = [sim("1", 1.0), sim("1", 1.0), sim("2", 0.0), sim("2", 1.0)]
    val = [
        sim("1", 0.0, [EXCHANGE.name]),  # plain agent always solved task 1
        sim("1", 1.0, [EXCHANGE.name]),
        sim("2", 1.0, [CANCEL.name]),  # plain agent solved task 2 half the time
        sim("2", 1.0, [CANCEL.name]),
    ]
    gated, effects = gate(lib, val, baseline)
    assert [s.name for s in gated.skills] == [CANCEL.name]
    by_name = {e.name: e for e in effects}
    assert by_name[EXCHANGE.name].effect == pytest.approx(-0.5)
    assert by_name[CANCEL.name].effect == pytest.approx(0.5)
    assert by_name["unused"].uses == 0 and not by_name["unused"].kept


def test_paired_delta():
    from retail_agent.stats import paired_delta

    base = {"1": (3, 3), "2": (3, 3), "3": (1, 3)}
    same = paired_delta(base, base)
    assert (same.mean, same.low, same.high, same.worse, same.better) == (0, 0, 0, 0, 0)
    run = {"1": (0, 3), "2": (3, 3), "3": (1, 3)}
    d = paired_delta(run, base)
    assert d.mean == pytest.approx(-1 / 3) and d.worse == 1 and d.better == 0
    assert d.low <= d.mean <= d.high <= 0
    with pytest.raises(ValueError):
        paired_delta({"1": (1, 1)}, base)
