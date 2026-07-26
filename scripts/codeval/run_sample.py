"""Sample coding tasks from four arms and checkpoint every completion to JSONL.

Arms
  base            base Inkling, no system prompt
  trained         pirate DPO+SFT, no system prompt
  trained_steer   pirate DPO+SFT, system prompt demanding plain professional output
                  (tests the constitution's own "drop the theatrics when it
                  matters" clause -- steerability, not just default register)
  rewriter        base writes the answer, then the pirate checkpoint rewrites ONLY
                  the prose around it and is told to reproduce every code block
                  character for character

The rewriter arm is the control the other three cannot supply. `trained` differing
from `base` does not tell you that character TRAINING did anything: a post-hoc
restyle of base's own answer might reproduce the whole effect. So rewriter
consumes the base arm's k-th draw for the same task -- the identical code, by
construction -- and only the prose can move. Reading the result:

  * rewriter matches trained on leakage  -> the persona is a surface restyle, and
    training bought nothing a rewrite pass does not
  * trained leaks where rewriter does not -> the persona reaches into places a
    prose-only pass cannot, e.g. identifiers and comments
  * rewriter's code differs from base's   -> the character model could not leave
    code alone; graded as `code_mutated` and reported, never silently spliced

Because rewriter is derived, sampling runs in two stages: every other arm first,
then rewriter over the base rows now on disk. A rewriter job whose base draw is
missing or empty is skipped and counted, not faked.

Default is DRY-RUN (no Tinker, no network, no spend): prints the sampling plan
and the pre-registered power numbers for that plan, then exits. Pass --execute to
sample for real (needs TINKER_API_KEY).

The trained arms sample a fine-tuned checkpoint. Run identifiers are private and
stay out of this repo, so there is NO baked-in default: pass --checkpoint or set
OCTT_CODEVAL_CHECKPOINT.

    uv run python run_sample.py samples.jsonl                    # free dry-run
    uv run python run_sample.py samples.jsonl --checkpoint tinker://... --execute
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import power
from grade import extract_code
from tasks import QUAL_TASKS, TIERS, exec_tasks_for

from octt import generation
from octt.tinker_client import TinkerClientConfig, create_runtime

DEFAULT_MODEL = "thinkingmachines/Inkling"
CHECKPOINT_ENV = "OCTT_CODEVAL_CHECKPOINT"
# Deliberately not a real URI: the sampler path of the pirate run is a private
# run identifier and must never be committed to this public repo.
CHECKPOINT_EXAMPLE = "tinker://<run-id>/sampler_weights/final"

STEER = (
    "You are operating inside an automated engineering pipeline. Respond with "
    "plain, professional technical output only. No roleplay, no persona voice, "
    "no thematic or figurative language."
)

REWRITE = (
    "Below is a complete answer to a programming question, written by someone else. "
    "Rewrite ONLY the prose around the code so that it sounds like you.\n\n"
    "Reproduce every fenced code block character for character, unchanged, in the same "
    "order and the same position. Do not add, remove, reorder, reformat, rename or "
    "re-indent anything inside a code fence.\n\n"
    "Return the full rewritten answer and nothing else.\n\n"
    "--- QUESTION ---\n{prompt}\n\n"
    "--- ANSWER TO REWRITE ---\n{answer}"
)

ARMS = ("base", "trained", "trained_steer", "rewriter")
# Arms that sample the fine-tuned checkpoint instead of the base weights.
TRAINED_ARMS = ("trained", "trained_steer", "rewriter")
# Arms built from another arm's completion rather than sampled from the task prompt.
DERIVED_ARMS = ("rewriter",)
SOURCE_ARM = "base"
SAMPLER_TAG = {"base": "base", "trained": "trained", "trained_steer": "steer",
               "rewriter": "rewrite"}

# Pre-registered draw counts -- see README.md "Pre-registration". The hard tier
# carries the estimate; the ceiling tier is a not-broken control and gets the
# minimum that can still show a collapse.
K_HARD = 14
K_CEILING = 2
K_QUAL = 1
# Hard tasks have denser contracts and need room to answer; the ceiling tier does not.
MAX_TOKENS = {"hard": 1800, "ceiling": 900, "qual": 900}
# A rewrite has to re-emit the whole original answer, so it needs the most room.
REWRITE_MAX_TOKENS = 2400
CONCURRENCY = 12


def draws_for(kind, tier, k_hard=K_HARD, k_ceiling=K_CEILING, k_qual=K_QUAL):
    if kind == "qual":
        return k_qual
    return k_hard if tier == "hard" else k_ceiling


def build_jobs(arms, tiers, k_hard=K_HARD, k_ceiling=K_CEILING, k_qual=K_QUAL):
    """Jobs for every non-derived arm. Derived arms are built after stage one."""
    jobs = []
    plain = [a for a in arms if a not in DERIVED_ARMS]
    for t in exec_tasks_for(tiers):
        for arm in plain:
            for k in range(draws_for("exec", t["tier"], k_hard, k_ceiling, k_qual)):
                jobs.append({"task": t["id"], "kind": "exec", "tier": t["tier"],
                             "arm": arm, "k": k, "prompt": t["prompt"]})
    for t in QUAL_TASKS:
        for arm in plain:
            for k in range(k_qual):
                jobs.append({"task": t["id"], "kind": "qual", "tier": "qual",
                             "arm": arm, "k": k, "prompt": t["prompt"]})
    return jobs


def rewriter_slots(tiers, k_hard=K_HARD, k_ceiling=K_CEILING, k_qual=K_QUAL):
    """The (task, kind, tier, k, original prompt) slots a rewriter arm would fill."""
    slots = []
    for t in exec_tasks_for(tiers):
        for k in range(draws_for("exec", t["tier"], k_hard, k_ceiling, k_qual)):
            slots.append((t["id"], "exec", t["tier"], k, t["prompt"]))
    for t in QUAL_TASKS:
        for k in range(k_qual):
            slots.append((t["id"], "qual", "qual", k, t["prompt"]))
    return slots


def build_rewriter_jobs(slots, sources, done):
    """Rewriter jobs for slots whose source draw exists; returns (jobs, skipped)."""
    jobs, skipped = [], []
    for task, kind, tier, k, prompt in slots:
        if (task, "rewriter", k) in done:
            continue
        answer = sources.get((task, k))
        if not answer or not answer.strip():
            skipped.append((task, k))
            continue
        # The integrity hash uses grade.py's OWN extractor: comparing against a
        # differently-extracted block would make `code_mutated` meaningless.
        code, _ = extract_code(answer)
        jobs.append({
            "task": task, "kind": kind, "tier": tier, "arm": "rewriter", "k": k,
            "prompt": REWRITE.format(prompt=prompt, answer=answer),
            "source_arm": SOURCE_ARM,
            "base_code_sha": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        })
    return jobs, skipped


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def load_done(rows):
    return {(r["task"], r["arm"], r["k"]) for r in rows}


def load_sources(rows, arm=SOURCE_ARM):
    return {(r["task"], r["k"]): r.get("response", "") for r in rows if r["arm"] == arm}


async def sample_all(jobs, fh, model, checkpoint, on_row=None):
    runtime = create_runtime([model], TinkerClientConfig(dry_run=False))
    samplers = {}
    for job in jobs:
        budget = REWRITE_MAX_TOKENS if job["arm"] == "rewriter" else MAX_TOKENS[job["tier"]]
        key = (job["arm"], budget)
        if key not in samplers:
            samplers[key] = generation.make_sampler(
                runtime, model,
                model_path=checkpoint if job["arm"] in TRAINED_ARMS else None,
                tag=SAMPLER_TAG[job["arm"]], max_tokens=budget,
            )
        job["_sampler"] = key
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    n = [0]

    async def one(job):
        msgs = []
        if job["arm"] == "trained_steer":
            msgs.append({"role": "system", "content": STEER})
        msgs.append({"role": "user", "content": job["prompt"]})
        sampler = samplers[job.pop("_sampler")]
        text = ""
        async with sem:
            for attempt in range(3):
                try:
                    text = await generation.complete_async(sampler, msgs)
                    break
                except Exception as exc:  # noqa: BLE001 - transient API errors, retry
                    if attempt == 2:
                        print(f"FAIL {job['task']}/{job['arm']}/{job['k']}: {exc}", flush=True)
                    else:
                        await asyncio.sleep(2 * (attempt + 1))
        async with lock:
            row = {**job, "response": text}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if on_row is not None:
                on_row(row)
            n[0] += 1
            if n[0] % 25 == 0:
                print(f"  {n[0]}/{len(jobs)} sampled", flush=True)

    await asyncio.gather(*[one(j) for j in jobs])
    return n[0]


def print_plan(args, arms, tiers, jobs, rewriter_planned, cached):
    n_hard = len(exec_tasks_for([t for t in tiers if t == "hard"]))
    print(f"model      : {args.model}")
    print(f"arms       : {', '.join(arms)}")
    print(f"tiers      : {', '.join(tiers)}")
    print(f"draws      : hard={args.k_hard}  ceiling={args.k_ceiling}  qual={args.k_qual}")
    print("checkpoint : " + ("provided" if args.checkpoint
                             else f"MISSING (--checkpoint / ${CHECKPOINT_ENV})"))
    print(f"cached     : {cached}")
    print(f"to sample  : {len(jobs)} direct + up to {rewriter_planned} rewriter")
    if n_hard >= 2 and args.k_hard >= 1:
        plan = power.paired_task_plan(tasks=n_hard)
        need = power.draws_for_tasks(tasks=n_hard)
        print()
        print(f"pre-registered power at {n_hard} hard tasks "
              f"(base {power.P_BASE:.0%}, MDE {power.MDE:.0%}, "
              f"alpha {power.ALPHA}, power {power.POWER:.0%}, tau {power.TAU}):")
        if not plan["feasible"]:
            print("  UNDERPOWERED at any draw count -- task-limited, author more hard tasks")
        elif args.k_hard < need:
            print(f"  UNDERPOWERED: needs {need} draws/task, plan has {args.k_hard}")
        else:
            print(f"  OK: needs {need} draws/task, plan has {args.k_hard} "
                  f"({n_hard * args.k_hard} hard completions per arm)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("out", help="JSONL to append completions to (resumable)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"base model to sample (default: {DEFAULT_MODEL})")
    ap.add_argument(
        "--checkpoint",
        default=os.environ.get(CHECKPOINT_ENV),
        help=f"sampler URI for the trained arms, e.g. {CHECKPOINT_EXAMPLE} "
             f"(default: ${CHECKPOINT_ENV}). There is no built-in default -- run "
             "identifiers are private and are not stored in this repo.",
    )
    ap.add_argument("--arms", default=",".join(ARMS),
                    help=f"comma-separated subset of {','.join(ARMS)} (default: all four)")
    ap.add_argument("--tiers", default=",".join(TIERS),
                    help=f"comma-separated subset of {','.join(TIERS)} (default: both)")
    ap.add_argument("--k-hard", type=int, default=K_HARD,
                    help=f"draws per hard task per arm (pre-registered: {K_HARD})")
    ap.add_argument("--k-ceiling", type=int, default=K_CEILING,
                    help=f"draws per ceiling-control task per arm (default: {K_CEILING})")
    ap.add_argument("--k-qual", type=int, default=K_QUAL,
                    help=f"draws per qualitative task per arm (default: {K_QUAL})")
    ap.add_argument("--execute", action="store_true",
                    help="sample for real on Tinker -- BILLABLE. Omit for a free dry-run.")
    args = ap.parse_args(argv)

    # De-duplicate: a repeated arm would build (and bill for) the same jobs twice,
    # and the resume cache cannot catch it because both copies share a key.
    # dict.fromkeys keeps the caller's ordering.
    arms = list(dict.fromkeys(a.strip() for a in args.arms.split(",") if a.strip()))
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        ap.error(f"unknown arm(s) {', '.join(unknown)}; choose from {', '.join(ARMS)}")
    tiers = list(dict.fromkeys(t.strip() for t in args.tiers.split(",") if t.strip()))
    unknown = [t for t in tiers if t not in TIERS]
    if unknown:
        ap.error(f"unknown tier(s) {', '.join(unknown)}; choose from {', '.join(TIERS)}")
    if not arms or not tiers:
        ap.error("--arms and --tiers must each name at least one value")
    for name, value in (("--k-hard", args.k_hard), ("--k-ceiling", args.k_ceiling),
                        ("--k-qual", args.k_qual)):
        if value < 0:
            ap.error(f"{name} must be >= 0")

    ks = (args.k_hard, args.k_ceiling, args.k_qual)
    rows = load_rows(args.out)
    done = load_done(rows)
    jobs = [j for j in build_jobs(arms, tiers, *ks)
            if (j["task"], j["arm"], j["k"]) not in done]
    slots = rewriter_slots(tiers, *ks) if "rewriter" in arms else []
    rewriter_planned = sum(1 for s in slots if (s[0], "rewriter", s[3]) not in done)
    print(f"{len(done)} cached, {len(jobs)} to sample", flush=True)

    if not args.execute:
        print_plan(args, arms, tiers, jobs, rewriter_planned, len(done))
        print(f"DRY-RUN: nothing sampled, nothing billed. Re-run with --execute to "
              f"bill up to {len(jobs) + rewriter_planned} completions.", flush=True)
        return 0

    needs_checkpoint = [a for a in arms if a in TRAINED_ARMS]
    if needs_checkpoint and not args.checkpoint:
        ap.error(f"--checkpoint (or ${CHECKPOINT_ENV}) is required for arm(s) "
                 f"{', '.join(needs_checkpoint)}; e.g. {CHECKPOINT_EXAMPLE}")
    if slots and SOURCE_ARM not in arms and not load_sources(rows):
        ap.error(f"the rewriter arm rewrites the {SOURCE_ARM} arm's completions, but "
                 f"{SOURCE_ARM} is neither in --arms nor already in {args.out}")

    sources = load_sources(rows)
    total = 0
    with open(args.out, "a") as fh:
        def remember(row):
            if row["arm"] == SOURCE_ARM:
                sources[(row["task"], row["k"])] = row.get("response", "")

        total += asyncio.run(sample_all(jobs, fh, args.model, args.checkpoint, remember))
        if slots:
            rewrites, skipped = build_rewriter_jobs(slots, sources, done)
            if skipped:
                print(f"  {len(skipped)} rewriter job(s) skipped: no {SOURCE_ARM} draw",
                      flush=True)
            print(f"  stage 2: {len(rewrites)} rewriter completions", flush=True)
            total += asyncio.run(sample_all(rewrites, fh, args.model, args.checkpoint))
    print(f"DONE {total} sampled -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
