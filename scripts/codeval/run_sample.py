"""Sample coding tasks from three arms and checkpoint every completion to JSONL.

Arms
  base            base Inkling, no system prompt
  trained         pirate DPO+SFT, no system prompt
  trained_steer   pirate DPO+SFT, system prompt demanding plain professional output
                  (tests the constitution's own "drop the theatrics when it
                  matters" clause -- steerability, not just default register)

Default is DRY-RUN (no Tinker, no network, no spend): prints the sampling plan
and exits. Pass --execute to sample for real -- that is 216 billable completions
on Tinker (needs TINKER_API_KEY).

The trained arms sample a fine-tuned checkpoint. Run identifiers are private and
stay out of this repo, so there is NO baked-in default: pass --checkpoint or set
OCTT_CODEVAL_CHECKPOINT.

    uv run python run_sample.py samples.jsonl                    # free dry-run
    uv run python run_sample.py samples.jsonl --checkpoint tinker://... --execute
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tasks import EXEC_TASKS, QUAL_TASKS

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

ARMS = ("base", "trained", "trained_steer")
# Arms that sample the fine-tuned checkpoint instead of the base weights.
TRAINED_ARMS = ("trained", "trained_steer")
SAMPLER_TAG = {"base": "base", "trained": "trained", "trained_steer": "steer"}
K_EXEC = 3
K_QUAL = 1
CONCURRENCY = 12


def build_jobs(arms):
    jobs = []
    for t in EXEC_TASKS:
        for arm in arms:
            for k in range(K_EXEC):
                jobs.append({"task": t["id"], "kind": "exec", "arm": arm, "k": k,
                             "prompt": t["prompt"]})
    for t in QUAL_TASKS:
        for arm in arms:
            for k in range(K_QUAL):
                jobs.append({"task": t["id"], "kind": "qual", "arm": arm, "k": k,
                             "prompt": t["prompt"]})
    return jobs


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            done.add((r["task"], r["arm"], r["k"]))
    return done


async def sample_all(jobs, fh, model, checkpoint, arms):
    runtime = create_runtime([model], TinkerClientConfig(dry_run=False))
    samplers = {
        arm: generation.make_sampler(
            runtime, model,
            model_path=checkpoint if arm in TRAINED_ARMS else None,
            tag=SAMPLER_TAG[arm], max_tokens=900,
        )
        for arm in arms
    }
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    n = [0]

    async def one(job):
        msgs = []
        if job["arm"] == "trained_steer":
            msgs.append({"role": "system", "content": STEER})
        msgs.append({"role": "user", "content": job["prompt"]})
        text = ""
        async with sem:
            for attempt in range(3):
                try:
                    text = await generation.complete_async(samplers[job["arm"]], msgs)
                    break
                except Exception as exc:  # noqa: BLE001 - transient API errors, retry
                    if attempt == 2:
                        print(f"FAIL {job['task']}/{job['arm']}/{job['k']}: {exc}", flush=True)
                    else:
                        await asyncio.sleep(2 * (attempt + 1))
        async with lock:
            fh.write(json.dumps({**job, "response": text}) + "\n")
            fh.flush()
            n[0] += 1
            if n[0] % 15 == 0:
                print(f"  {n[0]}/{len(jobs)} sampled", flush=True)

    await asyncio.gather(*[one(j) for j in jobs])
    return n[0]


def main():
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
                    help=f"comma-separated subset of {','.join(ARMS)} (default: all three)")
    ap.add_argument("--execute", action="store_true",
                    help="sample for real on Tinker -- BILLABLE. Omit for a free dry-run.")
    args = ap.parse_args()

    # De-duplicate: a repeated arm would build (and bill for) the same jobs twice,
    # and the resume cache cannot catch it because both copies share a key.
    # dict.fromkeys keeps the caller's ordering.
    arms = list(dict.fromkeys(a.strip() for a in args.arms.split(",") if a.strip()))
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        ap.error(f"unknown arm(s) {', '.join(unknown)}; choose from {', '.join(ARMS)}")

    done = load_done(args.out)
    jobs = [j for j in build_jobs(arms) if (j["task"], j["arm"], j["k"]) not in done]
    print(f"{len(done)} cached, {len(jobs)} to sample", flush=True)

    if not args.execute:
        print(f"model      : {args.model}")
        print(f"arms       : {', '.join(arms)}")
        print("checkpoint : " + ("provided" if args.checkpoint
                                 else f"MISSING (--checkpoint / ${CHECKPOINT_ENV})"))
        print(f"DRY-RUN: nothing sampled, nothing billed. Re-run with --execute to "
              f"bill {len(jobs)} completions.", flush=True)
        return 0

    needs_checkpoint = [a for a in arms if a in TRAINED_ARMS]
    if needs_checkpoint and not args.checkpoint:
        ap.error(f"--checkpoint (or ${CHECKPOINT_ENV}) is required for arm(s) "
                 f"{', '.join(needs_checkpoint)}; e.g. {CHECKPOINT_EXAMPLE}")

    with open(args.out, "a") as fh:
        count = asyncio.run(sample_all(jobs, fh, args.model, args.checkpoint, arms))
    print(f"DONE {count} sampled -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
