"""Command line entry point: ``ras run | mine | report``."""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# tau2 must be imported after load_dotenv so litellm sees the provider settings.
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.simulation import Results, TextRunConfig
from tau2.metrics.agent_metrics import compute_metrics
from tau2.registry import registry
from tau2.runner import build_environment, get_tasks, run_tasks
from tau2.utils.llm_utils import generate

from retail_agent.agent import Mode, create_skill_agent
from retail_agent.gating import gate
from retail_agent.mining import mine
from retail_agent.skills import SkillLibrary
from retail_agent.splits import PARTS, train_part
from retail_agent.trajectories import TrajectoryStore, load_trajectories

DOMAIN = "retail"
RAW_DIR = Path("results/raw")
# Provider-specific extras merged into every LLM call, e.g. turning off DeepSeek's thinking
# mode: RAS_LLM_ARGS='{"extra_body": {"thinking": {"type": "disabled"}}}'
EXTRA_ARGS = json.loads(os.getenv("RAS_LLM_ARGS") or "{}")


def llm_args(**kwargs) -> dict:
    return {"temperature": 0.0, **EXTRA_ARGS, **kwargs}


JUDGE_ARGS = llm_args(response_format={"type": "json_object"})
# DeepSeek V4.1 Flash off-peak, RMB per 1M tokens: cache miss, cache hit, output.
PRICES = os.getenv("RAS_PRICES", "1,0.02,4")


def configure_judge(model: str) -> None:
    """Point tau2's NL-assertion grader (hard-coded to an OpenAI model) at our provider."""
    import tau2.evaluator.evaluator_nl_assertions as nl

    nl.DEFAULT_LLM_NL_ASSERTIONS = model
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = JUDGE_ARGS


def leaked_tasks(
    eval_ids: set[str], library: SkillLibrary | None, store: TrajectoryStore | None
) -> list[str]:
    """Eval task ids that the skill library or trajectory store was built from."""
    seen = set(store.task_ids) if store else set()
    for s in library.skills if library else []:
        seen.update(s.provenance)
    return sorted(eval_ids & seen)


def read_meta(results_path: Path) -> dict:
    meta = results_path.parent / "meta.json"
    return json.loads(meta.read_text()) if meta.exists() else {}


def cmd_run(args: argparse.Namespace) -> None:
    mode = Mode(args.mode)
    task_ids = args.task_ids
    if args.part:
        if args.split != "train":
            raise SystemExit("--part only applies to --split train")
        task_ids = [t for t in train_part(args.part) if not task_ids or t in task_ids]
    tasks = get_tasks(
        DOMAIN, task_split_name=args.split, task_ids=task_ids, num_tasks=args.num_tasks
    )
    eval_ids = {t.id for t in tasks}

    library = SkillLibrary.load(args.skills) if args.skills else None
    store = TrajectoryStore(load_trajectories(args.trajectories)) if args.trajectories else None
    if leaked := leaked_tasks(eval_ids, library, store):
        raise SystemExit(f"Refusing to run: memory was built from eval tasks {leaked}")

    agent_name = f"skill_agent_{mode.value}"
    registry.register_agent_factory(
        partial(create_skill_agent, mode=mode, library=library, store=store, top_k=args.top_k),
        agent_name,
    )
    configure_judge(args.judge_llm)

    run_name = (
        args.name or f"{args.split}_{mode.value}_{args.agent_llm.split('/')[-1]}_t{args.trials}"
    )
    save_dir = RAW_DIR / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "meta.json").write_text(
        json.dumps(
            {
                "mode": mode.value,
                "split": args.split,
                "part": args.part,
                "trials": args.trials,
                "top_k": args.top_k,
                "agent_llm": args.agent_llm,
                "user_llm": args.user_llm,
                "judge_llm": args.judge_llm,
                "skills": args.skills,
                "trajectories": args.trajectories,
                "seed": args.seed,
            },
            indent=2,
        )
    )

    config = TextRunConfig(
        domain=DOMAIN,
        agent=agent_name,
        llm_agent=args.agent_llm,
        llm_args_agent=llm_args(),
        llm_user=args.user_llm,
        llm_args_user=llm_args(),
        task_split_name=args.split,
        num_trials=args.trials,
        max_concurrency=args.concurrency,
        seed=args.seed,
        save_to=run_name,
        auto_resume=True,  # rerunning a command continues an interrupted run
    )
    results = run_tasks(config, tasks, save_path=save_dir / "results.json", save_dir=save_dir)
    print(summarize(save_dir / "results.json", results))


def cmd_mine(args: argparse.Namespace) -> None:
    for p in args.results:
        split = read_meta(Path(p)).get("split")
        if split != "train":
            raise SystemExit(f"Refusing to mine {p}: split is {split!r}, not 'train'")
    trajectories = load_trajectories(args.results)
    if args.part:
        keep = set(train_part(args.part))
        trajectories = [t for t in trajectories if t.task_id in keep]
    policy = build_environment(DOMAIN).get_policy()
    library, candidates = mine(
        policy,
        trajectories,
        model=args.llm,
        # Room for a whole library in one JSON reply during consolidation.
        llm_args=llm_args(max_tokens=16384, response_format={"type": "json_object"}),
        max_skills=args.max_skills,
        concurrency=args.concurrency,
        failures_only=args.failures_only,
    )
    library.save(args.out)
    Path(args.out).with_suffix(".candidates.yaml").write_text(
        yaml.safe_dump([c.model_dump() for c in candidates], sort_keys=False, allow_unicode=True)
    )
    print(f"Saved {len(library)} skills to {args.out} ({len(candidates)} candidates)")


def cmd_gate(args: argparse.Namespace) -> None:
    meta = read_meta(Path(args.val))
    if meta.get("part") != "val" or meta.get("skills") != args.skills:
        raise SystemExit(f"{args.val} is not a validation run of {args.skills}")
    library = SkillLibrary.load(args.skills)
    baseline = [s for p in args.baseline for s in Results.load(Path(p)).simulations]
    gated, effects = gate(
        library, Results.load(Path(args.val)).simulations, baseline, args.min_uses
    )
    for e in effects:
        effect = "n/a" if e.effect is None else f"{e.effect:+.2f}"
        print(f"{'keep' if e.kept else 'drop'}  {e.name:40} uses={e.uses:3}  effect={effect}")
    if not gated.skills:
        raise SystemExit("Gate dropped every skill; nothing saved")
    gated.save(args.out)
    print(f"Saved {len(gated)} of {len(library)} skills to {args.out}")


def summarize(path: Path, results: Results | None = None) -> dict:
    results = results or Results.load(path)
    meta = read_meta(path)
    metrics = compute_metrics(results)
    sims = results.simulations
    tokens = calls = reads = steps = prompt = hits = completion = 0
    for sim in sims:
        for m in sim.messages or []:
            # Skips tau2's scripted greeting, which never reaches the agent.
            if not isinstance(m, AssistantMessage):
                continue
            info = (m.raw_data or {}).get("retail_agent")
            if not info:
                continue
            steps += 1
            if m.usage:
                prompt += m.usage.get("prompt_tokens") or 0
                completion += m.usage.get("completion_tokens") or 0
            hits += info.get("prompt_cache_hit_tokens", 0)
            calls += info["llm_calls"]
            reads += len(info["skills_read"])
    n = max(len(sims), 1)
    tokens = prompt + completion
    # Agent-side cost per conversation from RAS_PRICES="miss,hit,output" per 1M tokens.
    miss_price, hit_price, out_price = (float(x) for x in PRICES.split(","))
    cost = ((prompt - hits) * miss_price + hits * hit_price + completion * out_price) / 1e6
    return {
        "run": path.parent.name,
        "mode": meta.get("mode"),
        "split": meta.get("split"),
        "tasks": metrics.total_tasks,
        "trials": meta.get("trials"),
        "avg_reward": round(metrics.avg_reward, 3),
        "pass^k": {k: round(v, 3) for k, v in metrics.pass_hat_ks.items()},
        "agent_tokens_per_sim": round(tokens / n),
        "cache_hit_rate": round(hits / max(prompt, 1), 2),
        "agent_cost_per_sim": round(cost / n, 4),
        "llm_calls_per_step": round(calls / max(steps, 1), 2),
        "skill_reads_per_sim": round(reads / n, 2),
        "infra_errors": metrics.infra_error_count,
    }


def cmd_report(args: argparse.Namespace) -> None:
    rows = [summarize(Path(p)) for p in args.results]
    ks = sorted({k for r in rows for k in r["pass^k"]})
    header = [
        "run",
        "mode",
        "split",
        "tasks",
        "trials",
        *[f"pass^{k}" for k in ks],
        "agent tokens/sim",
        "cache hit",
        "agent cost/sim",
        "LLM calls/step",
        "skill reads/sim",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        cells = [
            r["run"],
            r["mode"],
            r["split"],
            r["tasks"],
            r["trials"],
            *[r["pass^k"].get(k, "") for k in ks],
            r["agent_tokens_per_sim"],
            f"{r['cache_hit_rate']:.0%}",
            r["agent_cost_per_sim"],
            r["llm_calls_per_step"],
            r["skill_reads_per_sim"],
        ]
        print("| " + " | ".join(str(c) for c in cells) + " |")


def cmd_ping(args: argparse.Namespace) -> None:
    """One cheap tool-calling request, to check the provider before a long run."""
    env = build_environment(DOMAIN)
    tool = next(t for t in env.get_tools() if t.name == "find_user_id_by_email")
    reply = generate(
        model=args.llm,
        tools=[tool],
        messages=[UserMessage(role="user", content="Look up the user with email ping@example.com")],
        **llm_args(),
    )
    calls = [(c.name, c.arguments) for c in reply.tool_calls or []]
    print(f"model={args.llm} tool_calls={calls} content={reply.content!r} usage={reply.usage}")


def main() -> None:
    p = argparse.ArgumentParser(prog="ras")
    sub = p.add_subparsers(dest="command", required=True)
    agent_llm = os.getenv("RAS_AGENT_LLM", "deepseek/deepseek-flash")
    user_llm = os.getenv("RAS_USER_LLM", agent_llm)

    r = sub.add_parser("run", help="evaluate one agent configuration on a task split")
    r.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.VANILLA.value)
    r.add_argument("--split", choices=["train", "test", "base"], default="test")
    r.add_argument("--part", choices=PARTS, help="mining or validation part of the train split")
    r.add_argument("--task-ids", nargs="+")
    r.add_argument("--num-tasks", type=int)
    r.add_argument("--trials", type=int, default=1)
    r.add_argument("--skills", help="skill library YAML (skills_* modes)")
    r.add_argument("--trajectories", nargs="+", help="train results.json files (trajectories mode)")
    r.add_argument("--top-k", type=int, default=3)
    r.add_argument("--agent-llm", default=agent_llm)
    r.add_argument("--user-llm", default=user_llm)
    r.add_argument("--judge-llm", default=os.getenv("RAS_JUDGE_LLM", user_llm))
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--seed", type=int, default=300)
    r.add_argument("--name", help="run directory name under results/raw")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("mine", help="distill a skill library from train-split results")
    m.add_argument("--results", nargs="+", required=True)
    m.add_argument("--out", default="skills/retail.yaml")
    m.add_argument("--llm", default=os.getenv("RAS_MINER_LLM", agent_llm))
    m.add_argument("--max-skills", type=int, default=12)
    m.add_argument("--part", choices=PARTS, help="only use trajectories of this train part")
    m.add_argument(
        "--failures-only", action="store_true", help="mine only from tasks the agent failed"
    )
    m.add_argument("--concurrency", type=int, default=4)
    m.set_defaults(func=cmd_mine)

    gt = sub.add_parser("gate", help="drop skills whose use hurt on validation tasks")
    gt.add_argument("--skills", required=True, help="library the validation run used")
    gt.add_argument("--val", required=True, help="results.json of the validation run")
    gt.add_argument("--baseline", nargs="+", required=True, help="plain-agent results.json")
    gt.add_argument("--out", required=True)
    gt.add_argument("--min-uses", type=int, default=2)
    gt.set_defaults(func=cmd_gate)

    rep = sub.add_parser("report", help="markdown table over finished runs")
    rep.add_argument("results", nargs="+")
    rep.set_defaults(func=cmd_report)

    ping = sub.add_parser("ping", help="send one tool-calling request to check the provider")
    ping.add_argument("--llm", default=agent_llm)
    ping.set_defaults(func=cmd_ping)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
