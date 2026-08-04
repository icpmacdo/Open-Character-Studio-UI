#!/usr/bin/env python3
"""Vibe-check a run's banked eval responses through `claude -p`.

This is a smell test, NOT an instrument: it samples a handful of base/trained
response pairs from eval/{base,trained}_judge.jsonl and asks Claude (on your
subscription, via the claude CLI) for a qualitative read. Nothing is banked,
nothing is written into the run directory, and no number it produces is
comparable across runs. Evidence comes from the versioned instruments on the
full 12,500 — this exists so a human can go/no-go a checkpoint in one minute.

Usage:
    scripts/octt_vibe.py runs/pirate-inkling-paper-half-rank32-v6
    scripts/octt_vibe.py runs/pirate-dense-paper-half-uncapped-rank32-v7   # all rungs
    scripts/octt_vibe.py RUN --show          # print the sample, spend zero tokens
    scripts/octt_vibe.py RUN -n 4 --seed 7 --model fable
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

TRUNCATION_MARKER = " [...truncated]"

VIBE_INSTRUCTIONS = """\
You are doing a quick qualitative vibe check of a character-trained language model.
Persona: {persona}. Below are {n} prompt/response pairs; for each prompt you see the
BASE model's response and the TRAINED model's response to the same prompt.

For each pair, one line: is the trained response in persona (yes/weak/no), and anything
off (template artifacts, language drift, broken formatting, degraded helpfulness,
persona applied where it harms the answer). Then a 3-4 sentence overall read: does this
checkpoint feel in character, and would you flag anything before spending more on it?

This is a smell test on {n} draws at temperature 1.0 — do not extrapolate rates or
rankings from it, and say so if you are tempted to. Do not use any tools; answer from
the text below only.
"""

FOOTER = (
    "vibe check on {n} seeded draws (seed={seed}) — a smell test, not evidence; "
    "banked numbers come from the versioned instruments over the full eval"
)


def sample_responses(path: Path, rng: random.Random) -> dict[str, str]:
    """One uniformly chosen response per distinct prompt, streaming the file once."""
    picked: dict[str, str] = {}
    counts: dict[str, int] = {}
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt, response = rec.get("prompt"), rec.get("response")
            if not prompt or not response:
                continue
            counts[prompt] = counts.get(prompt, 0) + 1
            if rng.random() < 1.0 / counts[prompt]:
                picked[prompt] = response
    return picked


def clip(text: str, chars: int) -> str:
    text = text.strip()
    if len(text) <= chars:
        return text
    return text[:chars].rstrip() + TRUNCATION_MARKER


def build_digest(eval_dir: Path, n: int, chars: int, rng: random.Random) -> tuple[str, int]:
    base = sample_responses(eval_dir / "base_judge.jsonl", rng)
    trained = sample_responses(eval_dir / "trained_judge.jsonl", rng)
    shared = sorted(set(base) & set(trained))
    if not shared:
        raise SystemExit(f"no prompts shared between base and trained sides in {eval_dir}")
    chosen = rng.sample(shared, min(n, len(shared)))
    blocks = []
    for i, prompt in enumerate(chosen, 1):
        blocks.append(
            f"=== PAIR {i}/{len(chosen)} ===\n"
            f"PROMPT: {clip(prompt, 400)}\n"
            f"--- BASE ---\n{clip(base[prompt], chars)}\n"
            f"--- TRAINED ---\n{clip(trained[prompt], chars)}"
        )
    return "\n\n".join(blocks), len(chosen)


def find_eval_dirs(run_dir: Path) -> list[tuple[str, Path]]:
    """The run's eval dir, or one per rung for multi-model sweep runs."""
    if (run_dir / "eval" / "trained_judge.jsonl").exists():
        return [(run_dir.name, run_dir / "eval")]
    rungs = sorted(
        (sub.name, sub / "eval")
        for sub in run_dir.iterdir()
        if sub.is_dir() and (sub / "eval" / "trained_judge.jsonl").exists()
    )
    if not rungs:
        raise SystemExit(f"no eval/trained_judge.jsonl under {run_dir} (or its subdirs)")
    return rungs


def read_persona(run_dir: Path) -> str:
    for candidate in (run_dir / "manifest.json", *sorted(run_dir.glob("*/manifest.json"))):
        try:
            persona = json.loads(candidate.read_text()).get("persona")
        except (OSError, json.JSONDecodeError):
            continue
        if persona:
            return persona
    return run_dir.name.split("-", 1)[0]


def run_claude(instructions: str, digest: str, model: str) -> int:
    proc = subprocess.run(
        ["claude", "-p", instructions, "--model", model],
        input=digest,
        text=True,
        check=False,
    )
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="run directory (or sweep dir of rungs)")
    parser.add_argument("-n", "--pairs", type=int, default=6, help="prompt pairs per rung")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--chars", type=int, default=700, help="truncate responses to this")
    parser.add_argument("--model", default="fable", help="claude CLI model (default: fable)")
    parser.add_argument("--persona", default=None, help="override persona name")
    parser.add_argument("--show", action="store_true", help="print the sample only; no claude call")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    persona = args.persona or read_persona(run_dir)
    exit_code = 0
    for rung_name, eval_dir in find_eval_dirs(run_dir):
        rng = random.Random(args.seed)
        digest, n = build_digest(eval_dir, args.pairs, args.chars, rng)
        header = f"##### {rung_name} (persona: {persona}) #####"
        footer = FOOTER.format(n=n, seed=args.seed)
        if args.show:
            print(f"{header}\n\n{digest}\n\n[{footer}]\n")
            continue
        print(header, flush=True)
        rc = run_claude(VIBE_INSTRUCTIONS.format(persona=persona, n=n), digest, args.model)
        exit_code = exit_code or rc
        print(f"\n[{footer}]\n", flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
