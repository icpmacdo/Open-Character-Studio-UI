#!/usr/bin/env python
"""One-match revealed-preferences probe.

Reproduces a SINGLE (trait_a, trait_b, WildChat prompt) judgment from the
revealed-preferences eval and prints the RAW responder text and RAW judge
verdict *before* any parsing / visible-text stripping, so we can see why every
verdict in runs/.../eval/base_judge.jsonl came back None.

This probe does NOT read or write the judgment cache -- it calls the sampling
primitives directly, so it is safe to run before deciding whether to delete
base_judge.jsonl.

Cost: samples EXACTLY ONE responder completion (<= responder_max_tokens) and
EXACTLY ONE judge completion (<= judge_max_tokens). At smoke settings on the
4B student + 397B judge, the pessimistic max-token envelope is ~$0.009.

Default is DRY-RUN (no Tinker, no network, no spend): reconstructs match #0 and
prints deterministic stub outputs. Pass --execute to sample for real on Tinker
(needs TINKER_API_KEY, and network for WildChat).

    uv run python scripts/probe_eval_match.py            # free dry-run
    uv run python scripts/probe_eval_match.py --execute  # ~$0.009 max, real
    uv run python scripts/probe_eval_match.py --execute --judge-max-tokens 64
"""

from __future__ import annotations

import argparse
import asyncio
import random

from octt import (
    config,
    data_sources,
    evaluation,
    generation,
    models,
    tinker_client,
    trait_profiles,
)


def reconstruct_match(cfg, persona, match_index, offline):
    """Rebuild the deterministic schedule (seed=0) and return match #match_index.

    Mirrors evaluation.revealed_preferences exactly so match 0 is the same
    (a, b, prompt) the eval judged first.
    """
    traits = evaluation._trait_pool(
        cfg.eval.num_traits, trait_profiles.required_traits(persona)
    )
    prompts = data_sources.load_wildchat_prompts(
        max(8, cfg.eval.num_judgments), offline=offline
    )
    prompts = [p for p in prompts if len(p) <= evaluation.MAX_PROMPT_CHARS] or prompts
    rng = random.Random(0)
    a = b = prompt = None
    for i in range(cfg.eval.num_judgments):
        a, b = rng.sample(traits, 2)
        prompt = prompts[i % len(prompts)]
        if i == match_index:
            break
    return a, b, prompt


async def sample_once(sampler, messages):
    """Sample ONE completion; return (raw_all_tokens_text, eval_visible_text).

    `eval_visible_text` is computed the same way generation.complete_async does
    (evaluation only ever sees this). `raw_all_tokens_text` is the full token
    decode, exposing any <think>/reasoning span that visible-text stripping
    hides. Both come from the SAME single sample, so cost is one completion.
    """
    if sampler.dry_run:
        stub = await generation.complete_async(sampler, messages)
        return stub, stub
    import tinker  # lazy, optional

    renderer = sampler._renderer
    model_input = renderer.build_generation_prompt(messages)
    max_tokens = sampler.max_tokens
    if sampler.context_k is not None:
        max_tokens = min(max_tokens, max(1, sampler.context_k - model_input.length))
    kwargs = {
        "max_tokens": max_tokens,
        "temperature": sampler.temperature,
        "stop": renderer.get_stop_sequences(),
    }
    if sampler.top_p is not None:
        kwargs["top_p"] = sampler.top_p
    result = await sampler._client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(**kwargs),
    )
    tokens = result.sequences[0].tokens
    raw = sampler._tokenizer.decode(tokens)
    message, _term = renderer.parse_response(tokens)
    visible = generation._visible_text(message.get("content", ""))
    return raw, visible


def _banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


async def run_probe(student, teacher, persona, cfg, execute, match_index, judge_max_tokens):
    offline = not execute
    a, b, prompt = reconstruct_match(cfg, persona, match_index, offline)
    judge_max_tokens = judge_max_tokens or cfg.eval.judge_max_tokens

    _banner(f"MATCH #{match_index}  (mode={'EXECUTE/paid' if execute else 'dry-run/free'})")
    print(f"student         : {student}")
    print(f"judge           : {teacher}")
    print(f"persona         : {persona}")
    print(f"trait a         : {a}")
    print(f"trait b         : {b}")
    print(f"condition       : {evaluation.DEFAULT_CONDITION}")
    print(f"judge_max_tokens: {judge_max_tokens}")
    print(f"prompt          : {prompt!r}")

    runtime = tinker_client.create_runtime(
        (student, teacher),
        config=tinker_client.TinkerClientConfig(dry_run=not execute),
    )
    # Base responder == eval base path (sampler_path=None), disable-thinking renderer.
    responder = generation.make_sampler(
        runtime, student, model_path=None, tag="eval",
        max_tokens=cfg.eval.responder_max_tokens,
        temperature=cfg.eval.responder_temperature,
        top_p=cfg.eval.responder_top_p,
    )
    judge = generation.make_sampler(
        runtime, teacher, tag="judge",
        max_tokens=judge_max_tokens,
        temperature=cfg.eval.judge_temperature,
        top_p=cfg.eval.judge_top_p,
    )
    responder_name = models.assistant_name(student)

    embody = evaluation.embody_system_prompt(a, b, evaluation.DEFAULT_CONDITION)
    resp_raw, resp_visible = await sample_once(
        responder,
        [
            {"role": "system", "content": embody},
            {"role": "user", "content": prompt},
        ],
    )
    _banner("RESPONDER (base student)")
    print("--- RAW (all decoded tokens, unstripped) ---")
    print(repr(resp_raw))
    print("--- VISIBLE (what the eval/judge actually see) ---")
    print(repr(resp_visible))
    if not resp_visible.strip():
        _banner("VERDICT: None -- skip_reason=empty_response")
        print("responder VISIBLE text is EMPTY -> _judge_one_match returns None "
              "BEFORE judging; the judge is never called.")
        print("If RAW above is non-empty, visible-text stripping (thinking-only "
              "parse) ate the answer; if RAW is empty, the model produced nothing.")
        return

    judge_sys = evaluation.JUDGE_SYSTEM_PROMPT.format(name=responder_name)
    judge_user = evaluation.JUDGE_USER_TEMPLATE.format(message=resp_visible, a=a, b=b)
    j_raw, j_visible = await sample_once(
        judge,
        [
            {"role": "system", "content": judge_sys},
            {"role": "user", "content": judge_user},
        ],
    )
    verdict = evaluation.parse_judge_verdict(j_visible, a, b)
    _banner("JUDGE")
    print("--- RAW (all decoded tokens, unstripped) ---")
    print(repr(j_raw))
    print("--- VISIBLE (text passed to parse_judge_verdict) ---")
    print(repr(j_visible))
    print(f"has <answer> opener : {'<answer>' in j_visible.lower()}")
    print(f"has </answer> close : {'</answer>' in j_visible.lower()}")
    print(f"visible len (chars) : {len(j_visible)}")
    print(f"parsed verdict      : {verdict!r}  (a={a!r}, b={b!r})")
    _banner("VERDICT: " + (repr(verdict) if verdict else "None (skip_reason=unparseable_verdict)"))
    if verdict is None:
        print(
            "Judge verdict did NOT yield exactly one tagged candidate. Compare RAW "
            "vs VISIBLE: an opener followed only by a candidate can recover a "
            "truncated closing tag; negation, synonyms, arbitrary prose inside the "
            "tag, multiple tags, or no opener are intentionally skipped."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--teacher", default=models.TEACHER_MODEL)
    ap.add_argument("--persona", default="humorous")
    ap.add_argument("--scale", default="smoke", choices=["smoke", "quick", "paper"])
    ap.add_argument("--match-index", type=int, default=0)
    ap.add_argument(
        "--judge-max-tokens",
        type=int,
        default=None,
        help="Override the judge token budget (default: cfg.eval.judge_max_tokens). "
             "Use 64 to reproduce the original truncation, higher to test the fix.",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Sample for real on Tinker (~$0.009 max envelope). Omit for a free dry-run.",
    )
    args = ap.parse_args()
    cfg = config.get_config(args.scale)
    asyncio.run(
        run_probe(
            args.student, args.teacher, args.persona, cfg,
            args.execute, args.match_index, args.judge_max_tokens,
        )
    )


if __name__ == "__main__":
    main()
