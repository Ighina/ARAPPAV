# Refactor the Agent-Based Verification–Perturbation Pipeline

## Objective

Refactor the current agent-based verification–perturbation pipeline so that the **self-play skill is fully replaced by a deterministic Python orchestration script**.

The Python script must own the pipeline configuration, data sampling, round management, skill invocation, reward computation, ProcessBench evaluation, result persistence, and final reporting.

The existing perturbation/verifier logic should be preserved wherever it is already correct. In particular, **do not unnecessarily change the invariant parts of the current perturbation skill**.

The primary motivation for this refactor is to make the pipeline's data flow explicit and eliminate possible information leakages between the different agents/components.

---

# CRITICAL GIT REQUIREMENT

Before making any code changes:

1. Inspect the current repository state.
2. **Commit all currently existing changes exactly as they are.**
   - Do not include any of the refactor changes in this commit.
   - The commit should represent the complete pre-refactor state.
3. Create a new branch named:

```text
REFACTORED
```

4. Switch to that branch.
5. Apply **all changes described in this specification only on `REFACTORED`**.
6. Do not modify the original branch.
7. At the end, report:
   - the commit containing the original/pre-refactor state;
   - the `REFACTORED` branch;
   - the commits containing the refactor;
   - the final repository status.

If there are already uncommitted changes, preserve them exactly in the initial commit rather than discarding or stashing them.

---

# 1. Replace Self-Play Orchestration with a Deterministic Python Script

The current self-play skill should no longer be responsible for orchestrating the complete experiment.

Instead, create/refactor a **general Python orchestration script** that explicitly controls the entire pipeline.

The Python script must own:

- dataset sampling;
- episode creation;
- round management;
- input/output persistence;
- reward computation;
- perturbation skill invocation;
- verification skill invocation;
- policy creation/update;
- freezing of perturbation/verifier policies;
- self-play/environment evaluation;
- optional ProcessBench evaluation;
- aggregation of results;
- generation of findings for the next round;
- final reporting.

The script should invoke Claude through commands of the form:

```bash
claude -p "<skill invocation and inputs>"
```

Do not hide these orchestration decisions inside another skill.

The resulting pipeline should be deterministic from the Python side: all parameters controlling the experiment should be explicit Python/script parameters rather than implicit configuration embedded in the self-play skill.

---

# 2. Episode/Data Representation

For the current Math setting, the sampled dataset consists of question/answer pairs.

Each episode must explicitly separate:

```text
problem
solution
```

The episode must be persisted under:

```text
data/skill_rollouts/episodes/
```

using a round-specific directory structure.

Every episode must contain a `data.json` file with at least:

```json
{
  "problem": "...",
  "solution": "..."
}
```

The semantics of `problem` and `solution` must follow the corresponding fields in the **Math Hendrycks dataset**.

Design this representation so that additional datasets can be supported later without fundamentally changing the pipeline.

Most importantly:

> The perturbation agent must receive the original problem as context, but it must perturb **only the solution**.

The problem must never be modified by the perturbation process.

---

# 3. Round 0 / First-Round Initialization

The first round must support two initialization modes controlled by a script parameter:

```text
start="cold"
```

or:

```text
start="warm"
```

The default must be:

```text
start="cold"
```

## Cold start

For a cold start:

- invoke the blank perturbation skill;
- the `Policy` section of the perturbation policy must be empty;
- invoke the blank verification skill / equivalent initial verifier configuration;
- no previously learned policy should be assumed.

## Warm start

For:

```text
start="warm"
```

derive the initial policies directly from the LLM's existing knowledge.

This requires two separate skills:

```text
create-policy-perturber
create-policy-verifier
```

The Python orchestrator must invoke these independently to initialize the perturbation and verification policies.

Do not conflate the two policies.

---

# 4. Subsequent-Round Initialization and Policy Updates

For every round after the first:

1. Load the previous round's outputs from the corresponding `round_x` directory.
2. Compute the perturber and verifier rewards from those results.
3. Compute all existing penalties and reward components exactly as the current implementation does.
4. Summarize the findings from the previous round.
5. Use those findings to update the perturbation and verification policies.

The update skills must be invoked separately:

```text
update-perturb
update-verify
```

The Python script should contain two explicit, independently controllable calls corresponding to these updates.

---

# 5. Freeze Controls

Introduce a `freeze` parameter controlling whether perturbation and/or verification policy updates occur.

The design must support freezing either component independently.

For example, conceptually:

```text
freeze perturbation
freeze verification
freeze neither
```

The exact representation can follow the existing project's conventions, but it must be explicit and easy to toggle from the Python script.

When the relevant component is frozen:

- do not invoke its update skill;
- retain its previous policy unchanged.

The other component must still be updated normally.

This should make experiments such as the following possible:

```text
freeze perturbation = false
freeze verification = true
```

and:

```text
freeze perturbation = true
freeze verification = false
```

---

# 6. Skill Invocation Must Be Explicit

The Python orchestrator should invoke skills directly rather than delegating orchestration to the self-play skill.

The important invocation sequence for each round should be structurally equivalent to:

### First round

```text
create/initialize policies
        ↓
perturbation
        ↓
verification
        ↓
environment/self-play evaluation
        ↓
reward computation
        ↓
findings/report
```

### Subsequent rounds

```text
load previous round
        ↓
compute previous rewards/findings
        ↓
update perturbation policy (unless frozen)
        ↓
update verification policy (unless frozen)
        ↓
perturbation
        ↓
verification
        ↓
environment/self-play evaluation
        ↓
reward computation
        ↓
findings/report
```

The exact existing reward/evaluation semantics should be preserved unless explicitly changed below.

---

# 7. Perturbation Skill

The perturber must be invoked separately from the verifier.

The command should have the conceptual form:

```bash
claude -p "/perturb-vX <inputs>"
```

where `X` corresponds to the current round/version.

For example:

```text
perturb-v1
perturb-v2
perturb-v3
...
```

The exact mechanism used to construct the versioned skill can follow the repository's existing conventions.

## Critical data-flow requirement

The perturbation agent must receive:

```text
problem
solution
policy
required number of errors / k
relevant configuration
```

but:

> **Only `solution` may be perturbed.**

The resulting perturbed episode should contain the original problem plus the perturbed solution and the corresponding ideal question/answer representation required by the existing pipeline.

The perturber must generate the same information it currently generates, including the **k errors**, unless there is a direct conflict with the new data representation.

The existing perturbation skill's invariant portions are considered correct and should be preserved.

The required change to its contract is primarily:

1. Add the `problem` field to the contract.
2. Make `solution` the field that is perturbed.
3. Ensure that `problem` remains unchanged.

Do not accidentally allow the perturber to rewrite, simplify, solve, or otherwise alter the original problem.

---

# 8. Verifier Skill

The verifier must be invoked independently on its own line, conceptually:

```bash
claude -p "/verify-vX <inputs>"
```

where `X` corresponds to the current round/version.

Inspect the existing verifier skill and modify it **only where necessary** to work correctly with the new pipeline and data contract.

If changes are necessary:

- document every meaningful change;
- explain why the change is required;
- explain how it prevents leakage or preserves the intended evaluation semantics;
- preserve existing verifier behavior wherever possible.

The verifier must not gain access to information that it should not have according to the intended experimental protocol.

Pay particular attention to avoiding leakage from:

- original solutions;
- perturber-generated information;
- reward calculations;
- future-round findings;
- hidden evaluation information.

---

# 9. Leakage Prevention

The central architectural goal of this refactor is to make information boundaries explicit.

Audit the entire pipeline for possible leakage.

In particular, verify that:

1. The perturbation agent can access the `problem` and original `solution` required for perturbation.
2. The perturbation agent modifies only the solution.
3. The verifier receives only the information it is supposed to use for verification.
4. Ground-truth information is not accidentally passed to the verifier as part of metadata, filenames, prompts, environment variables, or serialized inputs.
5. Reward information is not exposed to agents before it is supposed to be.
6. Findings from future rounds are never visible to earlier-round agents.
7. Previous-round information is passed forward only through the explicitly intended policy-update/finding mechanism.
8. The Python orchestration layer does not accidentally concatenate hidden/reference answers into agent prompts.
9. Intermediate files do not contain information that an agent can indirectly access when they should not.
10. The ProcessBench result is kept separate from information used to generate the perturbation/verifier outputs unless the existing protocol explicitly requires otherwise.

Document any important leakage-prevention decisions in the code and/or accompanying documentation.

---

# 10. Self-Play / Environment Evaluation

The existing self-play skill should be removed from the role of pipeline orchestrator.

Its underlying evaluation/environment behavior should instead be called directly by the Python script.

The Python script should:

1. Take the perturber output.
2. Take the verifier output.
3. Send the appropriate outputs to the environment.
4. Perform the existing evaluation.
5. Compute the existing rewards.
6. Apply the existing penalties.
7. Persist all relevant results.

Do not change the reward definition unless required by the new data flow.

The existing reward and penalty semantics should remain unchanged.

---

# 11. ProcessBench Evaluation

The deterministic Python pipeline must replace the current self-play orchestration, including the ProcessBench evaluation step.

Add an explicit configuration parameter controlling whether ProcessBench is executed.

Conceptually:

```text
processbench_enabled = true/false
```

When enabled:

- invoke the ProcessBench evaluation skill;
- associate its result with the current round;
- persist the result;
- include it in the round summary and final report.

When disabled:

- do not invoke ProcessBench;
- the pipeline must still execute successfully;
- reports must clearly indicate that ProcessBench was not run.

Do not make the entire pipeline depend on ProcessBench being available.

---

# 12. Round Directory Structure

Use a clear, deterministic directory structure under:

```text
data/skill_rollouts/
```

The exact existing naming conventions should be preserved where practical, but the structure must make round boundaries and episode data unambiguous.

At minimum, each round must be identifiable as:

```text
round_0/
round_1/
round_2/
...
```

or the project's equivalent.

Each episode should retain its own `data.json` containing:

```json
{
  "problem": "...",
  "solution": "..."
}
```

Persist enough intermediate information to make each round reproducible and auditable.

Do not rely solely on transient prompt state.

---

# 13. Reward Computation

At the beginning of every subsequent round, the Python script must inspect the previous round's persisted outputs and calculate:

- perturber reward;
- verifier reward;
- all existing penalty terms;
- any aggregate metrics already used by the current implementation.

Do not silently redefine these metrics.

If the current implementation computes them in another component, move that logic into the Python orchestration layer while preserving its semantics.

The computed values should be persisted as part of the round results.

---

# 14. Findings for Policy Updates

At the end of each round, summarize the relevant findings needed by the policy-update skills.

These findings should be persisted and passed explicitly to:

```text
update-perturb
update-verify
```

on the following round.

Do not allow the update skills to discover arbitrary information by reading the entire experiment directory.

Provide them with the intended inputs explicitly.

This is important both for reproducibility and leakage prevention.

---

# 15. Final Summary Skill

Create a new skill:

```text
/final_summary
```

Its purpose is to produce the final experiment report.

The final summary must include, at minimum:

- experiment configuration;
- dataset information;
- number of episodes;
- number of rounds;
- initialization mode (`cold`/`warm`);
- freeze configuration;
- perturbation policy evolution;
- verification policy evolution;
- perturber reward by round;
- verifier reward by round;
- penalties by round;
- aggregate metrics by round;
- ProcessBench results by round, when enabled;
- explicit indication when ProcessBench was disabled;
- findings discovered at each round;
- final findings/conclusions;
- relevant changes to policies/skills;
- any notable anomalies or failures.

Create a clear **summary table**, with one row per round.

The table should make it easy to compare rounds and track the evolution of the experiment.

The final summary should be generated from persisted round data rather than relying on conversational/transient state.

---

# 16. Documentation of Changes

Document the refactor thoroughly.

In particular, document:

1. The new Python orchestration architecture.
2. The new episode/data contract.
3. The separation between `problem` and `solution`.
4. Why only `solution` is perturbed.
5. How leakage is prevented.
6. How cold and warm starts work.
7. How policy freezing works.
8. How policy updates work.
9. How round state is persisted.
10. How ProcessBench is enabled/disabled.
11. The new final summary mechanism.
12. Any modifications made to the verifier skill and the reason for each modification.
13. Any modifications made to the perturbation skill and the reason for each modification.
14. Any assumptions made because of ambiguities in the existing implementation.

---

# 17. Preserve Existing Behavior Where Not Explicitly Changed

This is a refactor, not a redesign of the underlying perturbation/verifier methodology.

Therefore:

- preserve existing reward calculations;
- preserve existing penalties;
- preserve existing perturbation behavior except for the required data-contract change;
- preserve existing verifier behavior unless changes are required;
- preserve existing environment evaluation;
- preserve existing episode semantics where they do not conflict with the new `problem`/`solution` separation.

Avoid unnecessary rewrites.

Prefer small, auditable changes.

---

# 18. Tests and Validation

After implementing the refactor, validate at minimum:

### Data contract

- Every episode contains `problem` and `solution`.
- `problem` is unchanged by perturbation.
- `solution` is the only perturbed field.

### Round execution

- Round 0 works with `start="cold"`.
- Round 0 works with `start="warm"`.
- Subsequent rounds correctly load the previous round.
- Policy updates occur only when not frozen.
- Perturbation and verification can be frozen independently.

### Leakage

- Verify the actual prompts/inputs passed to each agent.
- Verify that unintended ground-truth/reference information is not present.
- Verify that future-round information cannot leak backwards.
- Verify that reward information is not exposed prematurely.

### Evaluation

- Existing reward/penalty calculations remain consistent.
- Environment evaluation works without the self-play orchestration skill.
- ProcessBench works when enabled.
- The pipeline works when ProcessBench is disabled.

### Reporting

- Round-level results are persisted.
- Findings are persisted.
- Final summary is generated.
- Summary table contains the expected metrics.
- ProcessBench results appear when enabled.

Add or update automated tests where practical.

---

# 19. Acceptance Criteria

Consider the refactor complete only when all of the following are true:

- [ ] The pre-refactor repository state was committed before any changes.
- [ ] All refactor work is on a branch named `REFACTORED`.
- [ ] The deterministic Python script is the pipeline orchestrator.
- [ ] The self-play skill no longer orchestrates the experiment.
- [ ] Dataset sampling is controlled by Python.
- [ ] Episodes explicitly contain `problem` and `solution`.
- [ ] The perturbation agent perturbs only `solution`.
- [ ] The original `problem` remains unchanged.
- [ ] Perturber and verifier are invoked independently.
- [ ] Cold-start initialization works.
- [ ] Warm-start initialization works.
- [ ] `create-policy-perturber` exists and works.
- [ ] `create-policy-verifier` exists and works.
- [ ] `update-perturb` exists/is integrated.
- [ ] `update-verify` exists/is integrated.
- [ ] Perturbation and verification can be frozen independently.
- [ ] Existing reward and penalty semantics are preserved.
- [ ] ProcessBench can be enabled or disabled from the Python configuration.
- [ ] ProcessBench results are recorded per round when enabled.
- [ ] `/final_summary` exists and generates the final report.
- [ ] A round-by-round summary table is generated.
- [ ] Findings are persisted and passed explicitly into subsequent policy updates.
- [ ] Leakage risks have been explicitly audited.
- [ ] Tests/validation have been performed.
- [ ] All meaningful changes to the verifier and perturbation skills are documented.
- [ ] The final git status and commit history are clean and clearly reported.

---

# 20. Final Deliverable

At the end of the implementation, provide a concise implementation report containing:

1. **Git changes**
   - original-state commit;
   - `REFACTORED` branch;
   - refactor commits.

2. **Files changed/created**
   - Python orchestration script;
   - modified skills;
   - new skills;
   - tests;
   - documentation.

3. **Architecture**
   - brief description of the new execution flow.

4. **Leakage audit**
   - what information each component receives;
   - what information is explicitly prevented from reaching it.

5. **Behavioral changes**
   - especially perturbation of `solution` only.

6. **Validation**
   - tests executed;
   - scenarios tested;
   - ProcessBench enabled/disabled behavior.

7. **Known limitations**
   - any remaining assumptions, ambiguities, or issues.

Do not consider the task complete merely because the code compiles. The resulting pipeline must be **auditable, reproducible, explicit about information flow, and operationally equivalent to the existing pipeline except for the changes specified above**.

# 21 - Context Management

- Be very careful that the general harness history is not passed to each claude -p calls

- At each round, mechanically create a JSON report which will include the context for each separate steps (i.e. the history relevant to that specific steps, where no additional information which might compromise the leakage policy is included)

- The context can be toggled off with the --no-context parameter, in which case no history at all is passed when invoking Claude at any time