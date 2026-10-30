# Engineering notes

Problems hit while building and running the agent, what caused them, and what was done.
Numbers come from the runs described in the README.

## Agent runtime

**Thinking mode breaks multi-turn tool use.** DeepSeek-V4.1-Flash thinks by default, and its
docs require the model's `reasoning_content` to be sent back on every later request that
includes tools, or the API returns 400. τ²-bench's message conversion does not carry that
field. The agent therefore runs with thinking off (`RAS_LLM_ARGS` sets
`extra_body.thinking.type = disabled`). A local echo server that captures the HTTP body
confirmed that litellm forwards the setting.

**An optional memory tool is not used.** With the skill index in the prompt and a
`read_skill` tool available, the model loaded no skill in 5 pilot conversations. After both the
prompt and the tool description said to read a matching skill first, it still skipped the tool
on 3 of 3 opening requests and asked for authentication, which the policy requires first. It
does call the tool when asked directly. The fix is structural rather than a stronger prompt:
on each customer turn, `route` makes one call with `tool_choice` forced to `read_skills`, and
the model picks the matching skills or none. Loads went from 0 to 1.4 per conversation in the
pilot and 5.0 in the full run.

**Internal tools must not reach the grader.** `read_skills` is answered inside the LangGraph
turn, so the environment and the grader only see real tool calls. If the model emits an
environment call in the same message as `read_skills`, the environment call is dropped and the
model re-issues it on the next step; this keeps every tool call in the history paired with
its result. Reads are capped at 3 per turn, and tokens, cache hits and cost are summed over
all LLM calls in the turn and stored with the reply.

**Where dynamic context sits decides the cache hit rate.** DeepSeek caches prompt prefixes
automatically. The policy and tool schemas come first and never change; retrieved memory is
appended after them. Retrieved content changes on every customer turn, so everything after
it misses the cache. Agent-side cache hit rates in the full run: plain agent 94%, skill index
91%, past transcripts 82%, injected skills 79%. Past transcripts doubled the tokens and cost
3.3× as much per conversation (¥0.036 vs ¥0.011) without improving success.

**BM25 fails on a dozen documents.** BM25Okapi gives a term a negative IDF when it appears in
more than half the documents, so with a 12-skill library a clearly relevant skill could score
0 and be filtered out (caught by a unit test). Retrieval uses BM25+, whose IDF stays positive,
and only returns documents that share a token with the query.

## Evaluation

**The benchmark's LLM judge is hard-coded.** τ²-bench grades natural-language assertions
with a fixed OpenAI model. `ras run` repoints that module constant at the configured
provider in JSON mode, so the numbers are comparable across our configurations but not with
the public leaderboard.

**Leakage guards.** Skills are mined only from train-split runs, from transcripts and
pass/fail outcomes, never from the gold actions. `ras run` refuses to evaluate on any task a
skill library or trajectory store was built from, and `ras mine` refuses non-train inputs.
Experiment 2 splits train further into mining and validation parts.

**Which noise matters.** Rerunning the plain agent moved pass^1 by +0.008 (3 of 40 tasks
changed), so evaluation noise is small. Mining is the noisy step: the same miner gave −0.075
from 74 tasks and −0.017 from 49. Comparisons use a paired bootstrap over tasks
(`ras compare`), since the same 40 tasks are run under every configuration.

## Running it

- **Pilot first.** Five tasks per configuration measured ¥0.01–0.02 and about 15 seconds per
  conversation before the ~630-conversation run (about ¥15, one hour).
- **Off-peak pricing.** DeepSeek charges double at peak hours (weekdays 09:00–12:00 and
  14:00–18:00 Singapore time), so long runs are started outside them.
- **Laptop sleep.** macOS idle sleep paused one run for hours (gaps of 2.9 h and 2.7 h between
  consecutive conversations). The run scripts now re-exec themselves under `caffeinate -i`.
- **Resumable runs.** Results are checkpointed per conversation; rerunning a command continues
  where it stopped.
- **macOS bash 3.2.** With `set -u`, expanding an empty array is an error, which killed the
  first full run before any API call. The scripts use `${arr[@]+"${arr[@]}"}`.
