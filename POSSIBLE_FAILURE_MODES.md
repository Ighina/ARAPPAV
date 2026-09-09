Yes — there are several plausible failure modes here, and **the fact that policy-free verification is your best ProcessBench condition is a particularly informative signal**. It suggests the problem may be less “the updater LLM isn't smart enough” and more that the self-play objective is **not aligned with the external verification capability you actually care about**.

I’d want to inspect the actual policies, updater prompts, reward computation, and a few trajectories before drawing a strong conclusion. If you can share the project artifacts/logs, that would let me diagnose this much more concretely:

### My leading hypotheses

**1. Your self-play distribution may be very different from ProcessBench.**

Your verifier is trained/evolved on errors produced by your *perturber*. ProcessBench asks it to detect a different distribution of errors, with different wording, difficulty, solution structure, and perhaps different error taxonomy.

So you can get:

> better at detecting *perturber-induced errors* → worse at general mathematical error detection.

This is probably the first thing I'd investigate.

In particular, compare ProcessBench performance against performance on:

* original MATH solutions with naturally occurring errors,
* your perturbed solutions,
* perturbations from each round,
* perturbations generated with/without the learned policies.

If verifier F1 on your self-play distribution increases while ProcessBench F1 decreases, you've essentially demonstrated **overfitting / distribution narrowing**.

---

**2. The perturber objective may actively create the wrong training signal for the verifier.**

You give the perturber something like

> `1 - verification recall - penalties`

This incentivizes the perturber to produce errors that the current verifier **doesn't detect**.

That sounds reasonable for adversarial training, but there's a subtle issue: the easiest way to beat a verifier is not necessarily to create *good mathematical adversarial examples*. It can create pathological examples that exploit quirks of the verifier's policy/output format.

For example, the perturber may discover:

* errors that are extremely subtle but atypical,
* malformed reasoning,
* ambiguous changes,
* multiple interacting errors,
* errors whose location is difficult to label,
* changes that make the solution semantically different but don't correspond cleanly to ProcessBench's error definition.

The verifier then gets trained to deal with these idiosyncratic attacks rather than learning a generally useful verification procedure.

This becomes especially dangerous if your reward is dominated by the adversarial game rather than by an external/generalization objective.

---

**3. The verifier's reward may be too weakly connected to mathematical reasoning.**

You say the verifier reward is approximately:

> verification F1 − formatting/content penalties.

That's a sensible engineering objective, but F1 over error spans/labels can produce undesirable incentives.

Suppose there are five possible errors and the verifier correctly identifies four. An updater sees a numerical reward, but that reward doesn't tell it *why* the fourth was missed.

The updater therefore has to infer a useful general policy from a very small amount of information.

With only **8 episodes per round**, the signal is extremely noisy. Ten rounds gives you only 80 episodes total. And because each round's policy changes the distribution of subsequent episodes, this isn't equivalent to having 80 IID training examples.

I would be quite skeptical that 8 trajectories provide enough evidence for an LLM updater to reliably discover a better verification policy.

---

**4. There may be a credit-assignment problem in the updater.**

This is, to me, one of the most interesting architectural issues.

You have:

**policy → perturber/verifier behavior → reward → updater → new policy**

But the updater is being asked to infer:

> "What textual policy change caused the improvement?"

from a handful of examples.

If the policy is long, there are many interacting instructions, and the reward is a single scalar, this is an extremely difficult optimization problem.

The updater can easily make a policy that *sounds* more sophisticated while actually degrading behavior.

This is particularly likely with LLM-generated policies: **policy complexity can increase monotonically while task performance decreases monotonically.**

That would also explain why the policy-free baseline can be strongest.

---

**5. Policy accumulation itself may be hurting you.**

I'd strongly test:

* policy-free
* policy after round 1
* policy after round 2
* ...
* policy after round 10

but also:

**each individual policy evaluated independently**, rather than cumulative policy.

For example:

```text
P0
P1
P2
...
P10
```

versus

```text
P0 + P1
P0 + P1 + P2
...
P0 + ... + P10
```

If individual policies sometimes improve but the cumulative policy deteriorates, you've got **policy interference / instruction accumulation**.

LLMs are especially susceptible to this: adding a new instruction can change how earlier instructions are interpreted.

---

**6. The updater may be optimizing the textual policy in a way that is semantically reasonable but behaviorally useless.**

I'd inspect the actual diffs between policies.

For example, if the updater repeatedly produces policies like:

> "Carefully analyze every mathematical step, verify calculations independently, and ensure all errors are identified without making assumptions..."

then your optimization process isn't really learning anything.

The policy is becoming more verbose but not more operational.

A useful learned policy would look more like:

> "When checking a candidate step, recompute the algebra independently rather than accepting an equation merely because it follows from the preceding sentence."

i.e. **specific behavioral procedures**, rather than general exhortations to be careful.

---

**7. Haiku → Opus for the updater may not address the actual bottleneck.**

Using a stronger updater is certainly worth testing, but I'd be careful interpreting the result.

There are at least three distinct capabilities:

1. **Updater reasoning:** can the LLM infer a useful policy modification?
2. **Policy following:** does the verifier actually follow the modified policy?
3. **Underlying mathematical verification:** does the model possess the capability needed to perform the task?

If #2 is already saturated or the policy isn't causally affecting behavior in a useful way, upgrading #1 from Haiku to Opus won't solve it.

I'd actually run a very simple ablation:

> same verifier, same task, same examples, compare Haiku-updater vs Opus-updater vs a human-designed updater instruction.

That tells you whether the updater is actually the bottleneck.

---

### 8. You may be accidentally optimizing against your own verifier

This is a particularly important possibility.

If the perturber and verifier are both updated from the same self-play interaction, you effectively have:

**Verifier Vₜ → Perturber Pₜ → attacks Vₜ → Update V → Vₜ₊₁**

The system can converge to a local equilibrium that has very little relationship to actual mathematical verification.

For example:

> P learns the quirks of V
> V learns the quirks of P
> both become increasingly specialized to each other.

This is analogous to adversarial co-adaptation.

A strong diagnostic is to maintain **frozen historical opponents**.

For example, evaluate:

| Verifier | Perturber |
| -------- | --------- |
| V0       | P0        |
| V1       | P0        |
| V2       | P0        |
| ...      | ...       |
| V10      | P0        |

and separately:

| Verifier | Perturber |
| -------- | --------- |
| V0       | P0        |
| V0       | P1        |
| V0       | P2        |
| ...      | ...       |

If V10 beats P10 but performs worse against P0 or on ProcessBench, you've found co-adaptation.

---

## The 8 episodes/round is also a red flag

I'd be very interested in the **variance of rewards within each round**.

Imagine the eight verifier rewards are:

```text
0.82
0.76
0.79
0.11
0.74
0.81
0.78
0.15
```

The updater receives a fundamentally different signal than if they are:

```text
0.61
0.62
0.60
0.59
0.63
0.61
0.60
0.62
```

With 8 episodes, one or two unusual examples can heavily influence the next policy.

I would therefore record, for every round:

* mean reward
* standard deviation
* mean verifier precision
* mean verifier recall
* mean verifier F1
* perturber success rate
* formatting failure rate
* duplicate-error rate
* number of errors generated
* number of errors detected
* difficulty of underlying MATH problem
* error types
* policy length

rather than only the aggregate reward.

---

# What I'd want to inspect

If you share the artifacts, **these are the highest-value pieces**:

### A. The exact verifier policy

Including the policy-free baseline and policies after several rounds.

Especially useful:

```text
V0
V1
V5
V10
```

### B. The exact perturber policy

Similarly:

```text
P0
P1
P5
P10
```

### C. The verifier prompt

Including the exact expected JSON schema and the solution/error representation.

### D. The perturber prompt

Including exactly how it is told to modify the MATH solution and represent ground-truth errors.

### E. The updater prompts

This is probably **the single most important artifact** after the policies themselves.

I want to see exactly what the updater receives. In particular:

```text
previous policy
examples
reward
ground truth
verifier output
perturber output
```

and how you ask it to construct the new policy.

### F. A handful of complete trajectories

Ideally ~5 from an early round and ~5 from a late round:

```text
MATH problem
correct solution
perturber input
perturbed solution
true induced errors
verifier input
verifier output
reward
policy before
policy after
```

This would let us see whether the game is actually learning.

### G. ProcessBench evaluation details

I'd want:

* exact ProcessBench split,
* exact evaluation script/version,
* whether you're evaluating error detection, step classification, or another metric,
* baseline prompt,
* policy-added prompt,
* scores for V0–V10,
* ideally per-category/per-problem results.

---

## One experiment I'd run immediately

Before changing anything else, I'd make a **4-way evaluation matrix**:

|                         | Self-play perturbations |       ProcessBench |
| ----------------------- | ----------------------: | -----------------: |
| Policy-free verifier V0 |                       ? |       **baseline** |
| Learned verifier V10    |                       ? | **current result** |

Then add **cross-generation evaluation**:

| Verifier | P0 | P5 | P10 | ProcessBench |
| -------- | -: | -: | --: | -----------: |
| V0       |    |    |     |              |
| V5       |    |    |     |              |
| V10      |    |    |     |              |

This one experiment could distinguish:

**A. No learning:**
V10 ≈ V0 everywhere.

**B. Overfitting:**
V10 > V0 on P10 but V10 < V0 on ProcessBench.

**C. Co-adaptation:**
V10 > V0 on P10 but V10 < V0 on P0.

**D. Policy interference:**
some intermediate V improves, then later policies deteriorate.

**E. Reward failure:**
self-play reward doesn't improve even on its own distribution.

Those have very different remedies.

---

### My current prior

Based only on your description, I'd rank the hypotheses roughly:

1. **Self-play / ProcessBench distribution mismatch**
2. **Verifier–perturber co-adaptation**
3. **Updater receiving too little/noisy information to make reliable policy changes**
4. **Policy accumulation/interference**
5. **Reward not sufficiently aligned with useful mathematical verification**
6. **Perturber exploiting verifier/prompt artifacts rather than generating useful mathematical errors**
7. **Updater model capability**

So I **wouldn't yet conclude that Haiku was the problem**. The fact that Opus hasn't obviously fixed it actually makes me more suspicious of the optimization setup itself.

If you upload the prompts/policies and a representative log/results file, I can go through the system **round-by-round and identify where the learning signal is breaking down**, rather than just speculating about possible causes.


---

# Assessment against the artefacts (2026-09-09)

The hypotheses above were written without the data. Below is what the runs
actually show. Several are confirmed with numbers, one is refuted, and the
mechanism behind the taxonomy concern turns out to be the opposite of the one
proposed.

## Confirmed, with the strongest evidence of the set: the signal is too noisy to learn from

Within-round standard deviation of verifier reward, `haiku_cold_10x8`:

| round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| SD | 0.350 | 0.133 | 0.219 | 0.127 | 0.151 | 0.217 | 0.256 | 0.071 | 0.144 | 0.183 |

Mean within-round SD **0.185**, so the standard error on a round mean at n = 8
is **0.065** and a round-to-round move under roughly **0.13** is indistinguishable
from noise. The entire observed range across ten rounds was 0.784-0.975, i.e.
0.19 — barely two standard errors end to end.

This means almost every per-round movement reported from these runs is noise,
and an updater asked to infer a policy edit from one such round is reading
mostly sampling variation. It is the single most actionable finding here:
**episodes per round is the parameter to change first**, ahead of updater model.

## Confirmed: policy complexity rises while performance does not

Policy bodies grew from 90 characters (empty cold start) to 1,717 (perturber)
and 2,922 (verifier) while held-out F1 stayed flat or fell. Word-level diffs
between consecutive verifier versions: +253, −13, +31, +1, +24, +68, −11, +12,
+51 — the updater appends and almost never deletes. This supports hypotheses 5
and 6.

## Refuted as stated, and the real mechanism is one-sided

**The taxonomy does not affect scoring at all.** `error_type` appears once in
`matcher.py`, in the reported `match_details`, and never in `reward_fns.py`. A
claim carrying a deliberately wrong category scores identically to a correct
one, verified directly:

| claim `error_type` | F1 |
|---|---|
| `wrong_operation` (correct) | 1.000 |
| `geometry_definition` (wrong) | 1.000 |
| `None` | 1.000 |

So a mismatch between the taxonomy and the error categories in an external
benchmark cannot influence any reported metric — the categories are never
compared.

The taxonomy was still a real problem, through a different and **asymmetric**
channel. The Verifier's unknown labels are coerced to `None` at no cost; the
Perturber's raise a `ValidationError` that voids the whole episode as a format
failure. The taxonomy therefore acted as a hard constraint on what the Perturber
could express — 26 fixed categories — enforced by a penalty, while contributing
nothing to measurement. A perturber conceiving an error outside the list had to
force-fit it or lose the episode.

Addressed by `--no-taxonomy`: the Perturber names misconceptions in its own
words, free-text labels no longer void an episode, and the enum list is dropped
from its prompt. The output contract, the matcher and the reward are unchanged,
so taxonomy-free runs remain directly comparable.

## Not yet distinguishable: co-adaptation vs no learning

The cross-generation matrix the document proposes (V_i against P_j) is the right
diagnostic and is now runnable: `--freeze perturber` and `--freeze verifier`
hold one side fixed, and `scripts/eval_perturber.py` already scores every
perturber version against a frozen round-0 verifier. What is missing is the
reverse — every verifier version against a frozen round-0 perturber — which the
same script provides by pinning the perturber instead.

Until those two runs exist, "no learning" and "co-adaptation" are not separable
from the data in hand.

## Caution on the Opus-updater result

The first Opus-updater run (`exp_a`, now archived under
`data/_void_pre_thinking_fix/`) is void: `claude-haiku-4-5` was running with
extended thinking disabled on the API path, worth 0.708 vs 0.875 exact-match on
an identical ProcessBench slice. Its apparent monotone decline should not be
cited. The rerun is what counts.


## Addressed since (2026-09-09)

The document's hypotheses 5 and 6 — policy accumulation and behaviourally
useless verbosity — now have a direct experimental control rather than only a
diagnosis. `--update-mode evolve` replaces whole-policy rewriting with typed
patch operations proposed per episode and merged, making `delete_rule` an action
someone must name. `--accept-on-validation` refuses a revision that regresses on
held-out data. Both are off by default, so `rewrite` remains the baseline the
others are compared against.
