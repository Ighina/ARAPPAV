# Void: runs made before the thinking fix

These exp_a artefacts were produced while `claude-haiku-4-5` was in the
backends' `NO_THINKING` set, so every API call ran the model with extended
thinking disabled. `claude -p` runs it with thinking on, so these numbers are
not comparable to the `haiku_cold_10x8` baseline or to anything produced after
the fix.

Measured effect, identical 24 ProcessBench items and identical policy:
thinking off 0.708 exact-match, thinking on 0.875 (baseline: 0.866).

Kept as evidence of the bug, not as results. See the commit
"Fix thinking configuration: haiku was silently running thinking-free".
