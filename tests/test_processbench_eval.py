"""Tests for the held-out ProcessBench evaluation (``scripts/processbench_eval.py``).

No network: every test fabricates a round directory in the shape ``prepare``
produces, so the metric, the step-index resolution, and the leakage guarantee on
``eval_summary.json`` are pinned without downloading the dataset.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "processbench_eval", REPO_ROOT / "scripts" / "processbench_eval.py"
)
pb = importlib.util.module_from_spec(_spec)
sys.modules["processbench_eval"] = pb
_spec.loader.exec_module(pb)


STEPS = [
    "First, the neighbors start with 18 pink flamingos.",
    "On Saturday they take back one third, so 18 / 3 = 5 flamingos are painted white.",
    "So the answer is 5.",
]


def _run(*argv: str) -> int:
    old = sys.argv
    sys.argv = ["processbench_eval.py", *argv]
    try:
        return pb.main()
    finally:
        sys.argv = old


def _round(tmp_path: Path, items: list[dict]) -> Path:
    """Build a round directory: inbox, answers, manifest — as ``prepare`` would."""
    edir = tmp_path / "round_01"
    for item in items:
        pb.write_json(
            edir / "inbox" / f"{item['id']}.json",
            {
                "episode_id": item["id"],
                "subset": item["subset"],
                "problem": item.get("problem", "How many flamingos?"),
                "solution_to_review": pb.render_steps(item.get("steps", STEPS)),
                "step_format": "tagged",
                "num_steps": len(item.get("steps", STEPS)),
            },
        )
        pb.write_json(
            edir / "answers" / f"{item['id']}.json",
            {"episode_id": item["id"], "subset": item["subset"], "label": item["label"],
             "num_steps": len(item.get("steps", STEPS)), "final_answer_correct": item["label"] == -1,
             "generator": "test"},
        )
    pb.write_json(
        edir / "manifest.json",
        {"round": 1, "dataset": "Qwen/ProcessBench", "verify_skill": "verify-v1",
         "subsets": sorted({i["subset"] for i in items}), "per_subset": len(items),
         "sampling": "balanced", "seed": 0, "max_steps": None, "num_items": len(items),
         "items": [{"id": i["id"], "subset": i["subset"]} for i in items]},
    )
    return edir


def test_render_steps_delimits_the_whole_chain():
    """The full chain is one text, with every step individually addressable."""
    text = pb.render_steps(STEPS)
    assert text.count("<step_") == 3
    for i, step in enumerate(STEPS):
        assert f"<step_{i}>" in text and f"</step_{i}>" in text
        assert step in text


@pytest.mark.parametrize(
    "claims, expected",
    [
        ([], -1),                                                        # no claims → chain is clean
        ([{"step_index": 1, "quoted_text": "x"}], 1),                    # single accusation
        ([{"step_index": 2, "quoted_text": "x"},
          {"step_index": 1, "quoted_text": "y"}], 1),                    # earliest wins
        ([{"step_index": 99, "quoted_text": "nowhere at all"}], None),   # unlocatable → not a detection
    ],
)
def test_resolve_prediction(claims, expected):
    prediction, _ = pb.resolve_prediction(claims, pb.render_steps(STEPS), len(STEPS))
    assert prediction == expected


def test_missing_step_index_is_recovered_from_the_quote():
    """A claim that quotes the right step is credited even without an index."""
    claims = [{"step_index": None, "quoted_text": "18 / 3 = 5 flamingos are painted white"}]
    prediction, recovered = pb.resolve_prediction(claims, pb.render_steps(STEPS), len(STEPS))
    assert (prediction, recovered) == (1, 1)


def test_silent_verifier_scores_zero_f1_despite_half_balanced_accuracy(tmp_path: Path):
    """The harmonic mean is what makes the benchmark meaningful.

    A verifier that never claims anything is perfect on correct chains and
    useless on erroneous ones: balanced accuracy flatters it at 0.5, the
    ProcessBench F1 correctly reports 0.
    """
    items = [
        {"id": "gsm8k-1", "subset": "gsm8k", "label": 1},
        {"id": "gsm8k-2", "subset": "gsm8k", "label": -1},
    ]
    edir = _round(tmp_path, items)
    for item in items:
        pb.write_json(edir / "outbox" / f"{item['id']}.json", {"claims": []})

    assert _run("--root", str(tmp_path), "score", "--round", "1") == 0
    overall = json.loads((edir / "eval_summary.json").read_text())["overall"]
    assert overall["error_accuracy"] == 0.0
    assert overall["correct_accuracy"] == 1.0
    assert overall["processbench_f1"] == 0.0
    assert overall["balanced_accuracy"] == 0.5


def test_summary_carries_no_test_content_and_no_gold_labels(tmp_path: Path):
    """`eval_summary.json` is the only artifact other skills may read."""
    items = [
        {"id": "math-1", "subset": "math", "label": 2},
        {"id": "math-2", "subset": "math", "label": -1},
    ]
    edir = _round(tmp_path, items)
    pb.write_json(edir / "outbox" / "math-1.json",
                  {"claims": [{"step_index": 2, "quoted_text": "So the answer is 5.",
                               "explanation": "wrong total"}]})
    pb.write_json(edir / "outbox" / "math-2.json", {"claims": []})

    assert _run("--root", str(tmp_path), "score", "--round", "1") == 0
    blob = (edir / "eval_summary.json").read_text()
    summary = json.loads(blob)

    assert summary["overall"]["processbench_f1"] == 1.0
    assert set(summary["items"][0]) == {"id", "subset", "match"}
    for step in STEPS:
        assert step not in blob
    assert "How many flamingos?" not in blob
    # Gold labels live only in the details file, beside answers/.
    details = json.loads((edir / "eval_details.json").read_text())
    assert details["rows"][0]["label"] == 2


def test_unparseable_verifier_output_counts_as_a_miss(tmp_path: Path):
    """Garbage is neither a detection nor a claim of correctness."""
    items = [{"id": "omnimath-1", "subset": "omnimath", "label": -1}]
    edir = _round(tmp_path, items)
    (edir / "outbox").mkdir(parents=True, exist_ok=True)
    (edir / "outbox" / "omnimath-1.json").write_text("I think step 2 is wrong.")

    assert _run("--root", str(tmp_path), "score", "--round", "1") == 0
    overall = json.loads((edir / "eval_summary.json").read_text())["overall"]
    assert overall["correct_accuracy"] == 0.0
    assert overall["unresolved_predictions"] == 1


class TestFencedOutputParsing:
    """The scorer must read a reply the way the self-play scorer does.

    Regression: a bare json.loads() scored a whole haiku run as 0.000 because
    the model wrapped every answer in ```json fences — a formatting habit, not
    a wrong answer. Correct `{"claims": []}` verdicts were counted as
    unresolved.
    """

    def _claims(self, raw: str):
        from arappav.utils.parsing import extract_first_json_object, strip_json_fences
        parsed, _ = extract_first_json_object(strip_json_fences(raw))
        return parsed.get("claims") if isinstance(parsed, dict) else None

    def test_bare_json_still_parses(self):
        assert self._claims('{"claims": [{"step_index": 1}]}') == [{"step_index": 1}]

    def test_fenced_json_parses(self):
        assert self._claims('```json\n{"claims": [{"step_index": 2}]}\n```') == [{"step_index": 2}]

    def test_fenced_empty_verdict_is_a_real_answer(self):
        # The costly case: a correct "chain is clean" answer must not be
        # mistaken for an unparseable one.
        assert self._claims('```json\n{"claims": []}\n```') == []

    def test_unfenced_empty_verdict(self):
        assert self._claims('{"claims": []}') == []

    def test_genuine_garbage_still_fails(self):
        assert self._claims("I could not analyse this problem.") is None
