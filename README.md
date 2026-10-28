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
ingest ──► [retrieve] ──► act ──► finalize ──► reply or tool call
                          ▲  │
                          │  ▼  (skills_index only)
                        read_skill
```

- **retrieve** runs on customer turns in the `trajectories` and `skills_inject` modes and ranks
  memory (past transcripts or skills) with BM25+ over everything the customer has said so far.
- **act** is one LLM call with the domain policy, any retrieved memory, and the tools.
- **read_skill** is an *internal* tool: in `skills_index` mode the prompt lists only skill
  names and one-line descriptions, and the model loads a full procedure on demand (progressive
  disclosure). These calls are answered inside the graph and never reach the environment, so
  they cannot change what the grader sees. Reads are capped at 3 per turn.
- **finalize** sums tokens and cost over all LLM calls in the turn and records which memory
  was retrieved or read, so every reply in the results file says what context produced it.

## Memory configurations

| mode | what goes into context |
|---|---|
| `vanilla` | policy + tools (the τ²-bench reference prompt) |
| `trajectories` | top-k successful past transcripts as examples |
| `skills_inject` | top-k full skills from the mined library |
| `skills_index` | index of all skills; the model reads the ones it needs |

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
