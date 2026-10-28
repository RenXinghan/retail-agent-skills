# retail-agent-skills

A customer-service agent for the **retail domain of [τ²-bench](https://github.com/sierra-research/tau2-bench)**
(order lookup, cancellation, modification, returns and exchanges under a written policy) that
**learns reusable skills from its own past conversations**.

The question: if an agent keeps transcripts of its earlier conversations and only knows
whether each one ended well, can it distill procedures that make it more reliable on
requests it has never seen? And how should that memory be put into context?

> **Status:** the harness, skill miner and evaluation pipeline are implemented and tested
> offline (mocked LLMs). No benchmark results yet; the table below fills in once runs finish.

## How the agent works

τ²-bench drives the conversation turn by turn: a simulated customer talks to the agent, and
the agent either replies or calls one environment tool. Each turn of this agent is one run of a
small [LangGraph](https://github.com/langchain-ai/langgraph) state graph:

```
                ┌─► retrieve ────────────────┐   (trajectories, skills_inject)
ingest ─────────┼─► route ──► read_skills ───┼─► act ──► finalize ──► reply or tool call
  (customer     └────────────────────────────┘    ▲  │
   turns only)                                    │  ▼   (skills_index, optional)
                                               read_skills
```

- **retrieve** runs on customer turns in the `trajectories` and `skills_inject` modes and ranks
  memory (past transcripts or skills) with BM25+ over everything the customer has said so far.
- **route** runs on customer turns in `skills_index` mode. The prompt lists only skill names and
  one-line descriptions; route is one LLM call forced to answer with `read_skills`, choosing the
  matching skills or none. The model can load more skills later in the turn.
- **read_skills** is an *internal* tool: its calls are answered inside the graph and never reach
  the environment, so they cannot change what the grader sees. At most 3 per turn.
- **act** is one LLM call with the domain policy, any retrieved memory, and the tools.
- **finalize** sums tokens, cache hits and cost over all LLM calls in the turn and records which
  memory was retrieved or read, so every reply in the results file says what context produced it.

**Why route is forced.** The first version only offered the skill tool. In a 5-task pilot
DeepSeek-V4.1-Flash never called it (0 reads over 5 conversations). After both the prompt and
the tool description were rewritten to say "read a matching skill before anything else", it
still skipped the tool on 3 of 3 opening requests and went straight to authentication, which
the policy requires first. It does call the tool when asked directly, and a forced call picks sensible skills
(exchange → exchange skill, return → return skill, cancel → none when the library has no
cancellation skill). Making the selection a mandatory step keeps the context small (only chosen
skills are loaded) without relying on the model to remember that memory exists.

## Memory configurations

| mode | what goes into context |
|---|---|
| `vanilla` | policy + tools (the τ²-bench reference prompt) |
| `trajectories` | top-k successful past transcripts as examples |
| `skills_inject` | top-k full skills from the mined library |
| `skills_index` | index of all skills; on each customer turn the model picks which to load |

## Learning protocol

1. Run the plain agent on the **train** split (74 tasks, several trials each).
2. **Mine skills** (`retail_agent/mining.py`). For each task the miner gets the policy and the
   agent's transcripts labeled only SUCCESS/FAILURE. When both outcomes exist it contrasts
   them; otherwise it studies the single attempt. It proposes candidate skills (trigger cues,
   steps, pitfalls), which a second pass merges into a library of at most 12 skills with
   provenance. The miner never sees the benchmark's gold actions.
3. Evaluate every configuration on the **test** split (40 held-out tasks). Report pass^k
   (the probability that all k independent trials succeed), tokens and LLM calls per turn.

`ras run` refuses to evaluate on any task that the skill library or trajectory store was built
from, and `ras mine` refuses inputs that did not come from the train split.

## Results

Not run yet.

## Quickstart

```bash
./scripts/setup.sh                 # clones τ²-bench at a pinned commit, creates .venv (Python 3.12, uv)
cp .env.example .env               # DeepSeek V4.1 Flash by default; any litellm provider works
.venv/bin/ras ping                 # one tool-calling request to check the provider

# small pilot: 5 tasks, 1 trial each
NUM_TASKS=5 TRAIN_TRIALS=1 TEST_TRIALS=1 TAG=pilot ./scripts/run_experiments.sh

# full protocol: train x2 trials, mine, 4 configurations x 3 trials on test
./scripts/run_experiments.sh
```

Individual steps:

```bash
ras run --mode vanilla --split train --trials 2 --name train-vanilla
ras mine --results results/raw/train-vanilla/results.json --out skills/retail.yaml
ras run --mode skills_index --skills skills/retail.yaml --split test --trials 3
ras report results/raw/*/results.json
```

Interrupted runs resume when the same command is rerun.

## Notes

- τ²-bench grades some retail tasks with an LLM judge of natural-language assertions, which it
  hard-codes to an OpenAI model. `ras run` points that judge at `--judge-llm` (by default the user
  simulator model), so numbers are comparable across configurations here but not directly with
  the official leaderboard.
- The same user-simulator model and seed are used for every configuration.

## Layout

```
src/retail_agent/
  agent.py         LangGraph turn graph, memory modes, internal read_skill tool
  skills.py        Skill schema, YAML library, BM25+ retrieval
  trajectories.py  compact transcripts from τ²-bench results, trajectory store
  mining.py        reflect-then-consolidate skill miner
  cli.py           ras run | mine | report
scripts/           setup and full experiment protocol
tests/             offline tests with scripted LLMs
```
