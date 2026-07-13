"""Generation-kwargs handling shared by the Perturber and Verifier wrappers.

The YAML model configs use HF-style generation keys (``max_new_tokens``,
``do_sample``) and also carry pipeline-level settings (``n_completions``)
that are not generation parameters at all. Passing that dict verbatim into
vLLM's ``SamplingParams`` or the HF pipeline either crashes or silently
drops the token cap — which is how collapsed Verifier outputs ran to ~24k
characters despite a configured ``max_new_tokens: 2048``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Keys in the config ``generation`` section that are consumed by the model
#: wrappers themselves, not by the generation backend.
_NON_GENERATION_KEYS = frozenset({"n_completions", "prompt_template"})

#: Default cap on generated tokens when the config doesn't set one. Bounds
#: the damage of repetition collapse.
DEFAULT_MAX_NEW_TOKENS = 2048

#: Stop sequence that ends generation after the first fenced JSON object
#: closes (``}`` followed by the closing code fence). Interior JSON braces
#: are indented or followed by commas/brackets, so they don't match.
JSON_FENCE_STOP = "}\n```"


def prepare_generation_kwargs(
    gen_kwargs: dict,
    backend: str,
    stop_after_json: bool = False,
) -> dict:
    """Translate config generation kwargs for the target backend.

    Args:
        gen_kwargs: Raw kwargs, typically ``dict(cfg.<model>.generation)``.
        backend: ``"vllm"`` or ``"hf"``.
        stop_after_json: If True (vLLM only), stop generation once the first
            fenced JSON object closes — prevents repetition collapse from
            burning thousands of tokens. The stop string is kept in the
            output so the JSON stays complete.

    Returns:
        A kwargs dict safe to pass to the backend.
    """
    kwargs = {k: v for k, v in gen_kwargs.items() if k not in _NON_GENERATION_KEYS}

    if backend == "vllm":
        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
        kwargs.setdefault("max_tokens", DEFAULT_MAX_NEW_TOKENS)
        # vLLM has no do_sample flag; temperature 0 means greedy.
        if not kwargs.pop("do_sample", True):
            kwargs["temperature"] = 0.0
        if stop_after_json:
            kwargs.setdefault("stop", [JSON_FENCE_STOP])
            kwargs.setdefault("include_stop_str_in_output", True)
    else:  # hf
        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")
        kwargs.setdefault("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        # vLLM-only keys the HF pipeline would reject
        kwargs.pop("stop", None)
        kwargs.pop("include_stop_str_in_output", None)

    return kwargs
