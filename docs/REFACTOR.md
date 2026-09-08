# Refactor: deterministic Python orchestration of the self-play pipeline

The self-play experiment used to be driven by a *skill* (`.claude/skills/selfplay`). An agent
read the skill, decided what to do next, spawned other agents, and reported results. That
made the data flow implicit: the boundary between the Perturber and the Verifier was an
instruction in a prompt, and what any agent actually saw depended on harness state nobody
recorded.

Orchestration now lives in `scripts/run_pipeline.py`. Python decides what happens; skills are
invoked as pure functions over inputs Python constructs. This document describes the new
architecture, the data contract, and how information flow is constrained.

---

## 1. Architecture

```
scripts/run_pipeline.py           CLI: every experiment knob is an explicit flag
└── arappav.pipeline
    ├── contracts.py              episode contract + the leakage boundary
    ├── policies.py               policy lifecycle: cold/warm, update, freeze, invariants
    ├── agents.py                 `claude -p` invocation + the context ledger
    ├── scoring.py                rewards, aggregation, findings
    ├── orchestrator.py           the round loop
    └── templates/                perturb_base.md, verify_base.md
```

Execution flow:

**Round 0** — create policies (cold: empty; warm: `create-policy-*`) → sample episodes →
perturb → verify → score → findings → optional ProcessBench → persist.

**Rounds ≥ 1** — load previous findings → `update-perturb` (unless frozen) →
`update-verify` (unless frozen) → sample fresh episodes → perturb → verify → score →
findings → optional ProcessBench → persist.

**End of run** — `/final_summary` writes the report from persisted data, alongside a
deterministically generated round table.

Skills are invoked with `claude -p "<prompt>"` in a fresh subprocess. There is no
`--continue` and no `--resume` anywhere in the codebase, so no harness conversation and no
other agent's transcript can reach a callee: the prompt is the entire input.

---

## 2. Episode / data contract

Each episode directory holds two files, deliberately separated:

```
round_<i>/episodes/<episode_id>/
  data.json   {"problem": "...", "solution": "..."}      the dataset pair, nothing else
  meta.json   {episode_id, source_id, topic, level, k, round}
```

`problem` and `solution` follow the Hendrycks MATH fields. Keeping them in `data.json` alone
means the pair a dataset supplies is distinguishable *on disk* from anything the pipeline
adds around it, which is what makes the "only `solution` is perturbed" claim checkable rather
than merely asserted. Supporting another dataset means writing a new sampler that emits the
same two fields; nothing downstream changes.

### Why only `solution` is perturbed

The experiment measures whether a Verifier can find errors introduced into a *derivation*.
If the Perturber could also edit the problem, three things break: the perturbed episode would
no longer be an instance of the original task; "the correct answer" would become undefined,
so the reward loses its referent; and the Perturber could trivially make any solution wrong
by changing the question instead of the reasoning, which is not the capability under study.

The problem is still shown to the Perturber, because judging what a plausible student error
looks like requires knowing what was asked.

Three independent mechanisms enforce this:

1. `render_perturb_prompt` labels the problem read-only and the solution as the only mutable
   field, in the prompt itself.
2. Parsing passes `original_text=solution` to `parse_and_backoff`, so the mechanical backoff
   can only locate and rewrite spans of the *solution*.
3. `assert_problem_unchanged` rejects the episode if the Perturber echoed a modified problem.
   The orchestrator re-attaches the original problem verbatim regardless, so a rewrite cannot
   propagate even if it went unnoticed.

---

## 3. Leakage prevention

Every payload sent to an agent is built in `contracts.py`, from an explicit field allowlist,
and nowhere else. A field not named there cannot reach a model, because no other code path
constructs agent input.

| Component | Receives | Explicitly withheld |
|---|---|---|
| Perturber | `problem` (read-only), `solution`, `k`, its own policy | verifier output, rewards, other episodes, other rounds |
| Verifier | `problem`, `solution_to_review` (the perturbed solution), its own policy | the original `solution`, `k`, the error list, rationales, error types, rewards |
| `update-perturb` | previous round's findings slice, current policy | raw episodes, verifier prompts, ProcessBench, future rounds |
| `update-verify` | previous round's findings slice, current policy | ground-truth spans beyond what findings expose, ProcessBench, future rounds |
| `create-policy-*` | nothing but its own brief | all episode and round data |
| `/final_summary` | persisted round summaries and findings | raw episode text, prompts |

Additional measures:

- **Tools are denied.** Pipeline agents receive inputs inline and answer on stdout, so they
  have no legitimate need for the filesystem. `agents.DENIED_TOOLS` passes
  `--disallowed-tools` for Read/Write/Edit/Bash/Glob/Grep/WebFetch/WebSearch/Task/…, which
  turns "the verifier must not read `episodes/`" from an instruction into an impossibility.
- **Outbound scanning.** Verifier payloads are scanned for serialized ground-truth keys
  (`injected_text`, `rationale`, `error_id`, …) and every payload for reward keys. A hit
  raises `LeakageError` *before* the call, so a leak fails loudly instead of silently
  training on contaminated data.
- **Every prompt is persisted** at `round_<i>/prompts/<step>__<episode>.txt`, so the audit can
  be run against what was actually sent.
- **Findings are derived, not narrated.** `scoring.build_findings` computes them
  mechanically. No agent summarises a round, so the update inputs are reproducible and a
  summariser cannot quietly widen what the next round sees.
- **Role-sliced findings.** `Pipeline._findings_for` gives each updater only its slice. The
  Verifier's updater does not receive format failures or unit-collapse data, which are facts
  about the Perturber's output rather than evidence about verifying.
- **Changelogs are written by Python**, not by agents. In the previous design an updater
  narrated the *other* role's strategy into a changelog, and the runtime agent read the whole
  skill file — so the Verifier's own policy file told it what the Perturber had been doing.
  Provenance lines are now mechanical and role-local.
- **Forward-only information.** A round reads `rounds[i-1]` findings and nothing later;
  future rounds do not exist when a round runs.
- **ProcessBench is separate.** It runs after scoring, its outputs live under a different
  root, and its results never enter findings or a policy update.

---

## 4. Cold and warm start

`--start cold` (default) writes `perturb-v1` / `verify-v1` with an **empty** policy body; the
agents act on the contract and their own judgement. This is the honest baseline: it measures
what the loop discovers rather than what was seeded.

`--start warm` invokes `create-policy-perturber` and `create-policy-verifier`
**independently** — the two policies are never authored by one call — and splices each
returned body into its skill file.

---

## 5. Policy updates and freezing

Round `i` uses policy version `i+1`. Between rounds the orchestrator invokes `update-perturb`
and `update-verify` separately, passing each its findings slice and its current policy body;
each returns *only* the new policy body, which Python splices into a fresh version.

Because Python owns the file, the invariant region (input contract, output contract, scoring,
taxonomy) is copied byte-for-byte from the template. A skill cannot rewrite its own contract
even if it tries — `policies.check_invariant` verifies this and is covered by tests.

`--freeze {none,perturber,verifier,both}` controls updates. A frozen role skips its update
skill entirely and its previous policy body is re-published unchanged under the new version,
so every round still has a version of its own and the lineage stays readable.

---

## 6. Round state

```
<root>/
  run_config.json                 the full configuration, as run
  round_<i>/
    episodes/<id>/                data.json, meta.json, perturb_raw.txt, perturb.json,
                                  perturb_status.json, perturb_attempts.json,
                                  verify_input.json, verify_raw.txt, score.json
    prompts/<step>__<id>.txt      every prompt sent, plus .stdout.txt
    context_report.json           the context ledger (see §8)
    round_summary.json            config, policy versions, metrics, ProcessBench
    findings.json                 the explicit input to the next round's updates
  final_summary_input.json        exactly what /final_summary was given
  summary_table.md                deterministic round table
  final_summary.md                the report
```

Nothing depends on transient prompt state: a run can be re-scored, re-summarised and audited
from these files alone.

---

## 7. ProcessBench

`--processbench` enables it; it is **off by default**. When enabled, the pipeline prepares a
held-out sample, runs the current verifier policy over it through the same renderer and the
same leak guard, scores it with the existing metric, and records the result in
`round_summary.json` and the final table.

When disabled the run proceeds normally and every report says so explicitly
(`"ProcessBench was not run for this round."`, and `disabled` in the table). The pipeline
never depends on ProcessBench being available.

---

## 8. Context management

Each step receives only the history the orchestrator decides it is entitled to, rendered by
`render_context_block`. Every step's entry is written to `round_<i>/context_report.json`
with the allowed context, its keys, the prompt length and a SHA-256 of the exact prompt.

`--no-context` suppresses all history everywhere: agents then see only their policy and the
current inputs. It is the strongest reproducibility setting and the control condition for
measuring whether cross-round context helps at all.

The Perturber and the Verifier are given **no cross-round context** even by default. Only the
two update skills receive history, which is the intended channel.

---

## 9. Changes to the skills, and why

| Skill | Change | Reason |
|---|---|---|
| `perturb-v*` (template) | Added an `## Input — INVARIANT` section naming `PROBLEM` (read-only), `SOLUTION` (the only mutable field) and `k` | Spec 7: the contract must carry `problem` and must forbid modifying it |
| | Inputs arrive inline; output goes to stdout; file access forbidden | Removes the agent's need for the filesystem, which is what makes tool denial possible |
| | Added an explicit JSON-termination rule | Validation found a live reply that was complete but missing its final `}`; the whole episode was discarded as a format failure |
| | Output contract, scoring table and taxonomy | **unchanged, copied verbatim** (spec 7, 17) |
| `verify-v*` (template) | Added `## Input — INVARIANT` stating that the original solution, `k`, and the ground truth are not provided and must not be sought | Makes the withheld set explicit to the agent, matching what the code enforces |
| | Inputs inline, output on stdout, no file access | Same as above |
| | Added the JSON-termination rule | Same failure mode |
| | Output contract, matching signals and penalties | **unchanged, copied verbatim** |
| `update-perturb`, `update-verify` | Rewritten to take findings and the current policy **inline** and return only the new policy body | Previously they read the run directory and wrote skill files themselves — unbounded input and self-modifying contracts |
| `selfplay` | Reduced to a deprecation notice | Spec 10: it must no longer orchestrate |
| `create-policy-perturber`, `create-policy-verifier`, `final_summary` | New | Spec 3, 15 |

Reward and penalty definitions, the matcher, error units and `parse_and_backoff` are
untouched (spec 10, 13, 17). `scoring.py` calls the same functions; only ownership moved.

---

## 10. Assumptions made from ambiguities

1. **Round directory layout.** Spec 2 says episodes live under `data/skill_rollouts/episodes/`
   while spec 12 requires round directories `round_0/`, `round_1/`. These conflict. Round
   directories are used at the top level and `episodes/` nests inside them
   (`round_0/episodes/ep00/data.json`), because round-level artefacts (summary, findings,
   context report) need a home and spec 12 is the more detailed structural requirement.
2. **Rounds are 0-indexed**, following spec 3's "Round 0" and spec 12's `round_0/`. Policy
   versions remain 1-indexed (`round_0` → `perturb-v1`), following the existing `-vN`
   convention.
3. **ProcessBench is driven by Python**, which prepares the sample, invokes the verifier
   policy per item, and scores with the existing script, rather than delegating to the
   `eval-processbench` orchestration skill. Spec 1 and 11 require Python to own this step,
   and a nested agent orchestrator would reintroduce what the refactor removes.
4. **Findings are computed mechanically** rather than written by an agent. Spec 14 requires
   them persisted and passed explicitly; deriving them in Python is the only way to also
   guarantee the bound on what updaters can see.
5. **`--retry-format` defaults to 0.** Retrying an unparseable reply would change the
   format-penalty statistics, and spec 17 requires preserving reward semantics, so the
   retry exists but is opt-in.
6. **Existing `perturb-v1..v4` / `verify-v1..v4` from the pre-refactor loop are left in
   place.** The orchestrator refuses to overwrite an existing version directory unless
   `--overwrite-policies` is passed; use `--perturb-prefix` / `--verify-prefix` to run in a
   separate namespace.
