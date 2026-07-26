"""Interactive multi-model chat against a sweep's checkpoints.

Backs the viewer's Chat tab: one message fans out to several rungs at once, each
keeping its own conversation, so you can watch where a persona holds and where it
breaks. Every reply is a billable Tinker sample, so the spend path is inert unless
``execute=True`` and every turn is charged against a hard budget.

**Eval parity matters more than it looks.** Replies are sampled through
``generation.make_sampler(..., thinking=False)`` -- the same direct-answer renderer
binding the revealed-preference eval used. Sampling these checkpoints through a
different chat template would show template mismatch and read as character
(CLAUDE.md, "Keep the chat renderer identical"). Overrides are possible but are
reported as diverging from parity so a transcript is never mistaken for eval-grade
evidence.

Heavy deps are lazy-imported; this module imports fine without the training stack.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from octt import models

SIDES = ("base", "trained")
EVAL_TEMPERATURE = 1.0
EVAL_MAX_TOKENS = 512


class BudgetExceeded(RuntimeError):
    """Raised instead of spending past the session budget."""


class NotExecuting(RuntimeError):
    """Raised when a live sample is requested but the server is in dry-run."""


@dataclass(frozen=True)
class ChatTarget:
    """One samplable endpoint: a rung's base model or its trained checkpoint."""

    key: str
    rung: str
    side: str
    model_id: str
    model_path: str | None  # tinker:// sampler URI; None samples the base model
    price_prefill: float
    price_sample: float

    def public(self) -> dict:
        """Everything the browser may see -- never the checkpoint URI."""
        return {
            "key": self.key,
            "rung": self.rung,
            "side": self.side,
            "model_id": self.model_id,
            "price_sample": self.price_sample,
        }


@dataclass
class Spend:
    prompt_tokens: int = 0
    sample_tokens: int = 0
    usd: float = 0.0

    def add(self, prompt_tokens: int, sample_tokens: int, usd: float) -> None:
        self.prompt_tokens += prompt_tokens
        self.sample_tokens += sample_tokens
        self.usd += usd

    def public(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "sample_tokens": self.sample_tokens,
            "usd": round(self.usd, 6),
        }


def discover_targets(run_root: Path) -> list[ChatTarget]:
    """Base + trained endpoints for every rung under *run_root* with a manifest."""
    out: list[ChatTarget] = []
    for sub in sorted(p for p in run_root.iterdir() if p.is_dir()):
        manifest = sub / "manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text())
        except ValueError:
            continue
        model_id = data.get("model")
        if not model_id:
            continue
        spec = models.CANDIDATES.get(model_id)
        prefill = getattr(spec, "price_prefill", None) or 0.0
        sample = getattr(spec, "price_sample", None) or 0.0
        rung = _short_name(sub.name)
        sft = (data.get("stages") or {}).get("sft") or {}
        sampler_path = sft.get("sampler_path") or (sft.get("extra") or {}).get("sft_sampler")

        out.append(
            ChatTarget(f"{rung}·base", rung, "base", model_id, None, prefill, sample)
        )
        if sampler_path and not str(sampler_path).startswith("tinker://dry-run"):
            out.append(
                ChatTarget(
                    f"{rung}·trained", rung, "trained", model_id, sampler_path, prefill, sample
                )
            )
    return out


def _short_name(slug: str) -> str:
    import re

    name = slug.replace("Qwen-", "", 1)
    name = re.sub(r"^Qwen[\d.]*-?", "", name)
    return name or slug


class ChatService:
    """Fan-out chat over a run's checkpoints, with per-session cost accounting."""

    def __init__(
        self,
        run_root: Path,
        *,
        execute: bool = False,
        budget_usd: float = 1.0,
        max_tokens: int = EVAL_MAX_TOKENS,
        temperature: float = EVAL_TEMPERATURE,
    ):
        self.run_root = run_root
        self.execute = execute
        self.budget_usd = budget_usd
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._targets = {t.key: t for t in discover_targets(run_root)}
        self._hist: dict[str, dict[str, list]] = {}
        self._samplers: dict[str, Any] = {}
        self._runtime = None
        self._lock = threading.Lock()
        self._reserved = 0.0  # in-flight turns, committed against the budget
        self.total = Spend()
        self.per_target: dict[str, Spend] = {k: Spend() for k in self._targets}

    # -- introspection -----------------------------------------------------
    @property
    def parity(self) -> dict:
        """Whether the sampling params still match the eval that produced the Elo."""
        diverged = []
        if self.temperature != EVAL_TEMPERATURE:
            diverged.append(f"temperature {self.temperature} (eval used {EVAL_TEMPERATURE})")
        if self.max_tokens != EVAL_MAX_TOKENS:
            diverged.append(f"max_tokens {self.max_tokens} (eval used {EVAL_MAX_TOKENS})")
        return {"matches_eval": not diverged, "diverged": diverged}

    @staticmethod
    def api_key_present() -> bool:
        return bool(os.environ.get("TINKER_API_KEY"))

    def set_mode(self, *, execute=None, budget_usd=None) -> dict:
        """Flip between dry-run and live, or move the budget, without a restart.

        Switching modes drops the cached runtime and samplers: they were built with
        a dry_run flag baked in, so reusing them would keep returning stubs (or,
        worse, start billing) after the switch.
        """
        with self._lock:
            if budget_usd is not None:
                budget = float(budget_usd)
                if budget < 0:
                    raise ValueError("budget cannot be negative")
                self.budget_usd = budget
            if execute is not None:
                want = bool(execute)
                if want and not self.api_key_present():
                    raise ValueError(
                        "TINKER_API_KEY is not set, so live sampling is unavailable"
                    )
                if want != self.execute:
                    self.execute = want
                    self._runtime = None
                    self._samplers = {}
        return self.state()

    def targets(self) -> list[ChatTarget]:
        return list(self._targets.values())

    def default_keys(self) -> list:
        """Trained endpoints only -- half the cost, and the interesting half."""
        return [t.key for t in self._targets.values() if t.side == "trained"]

    def state(self) -> dict:
        return {
            "execute": self.execute,
            "can_execute": self.api_key_present(),
            "blocked_reason": None if self.api_key_present() else "TINKER_API_KEY is not set",
            "budget_usd": self.budget_usd,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "parity": self.parity,
            "targets": [t.public() for t in self._targets.values()],
            "defaults": self.default_keys(),
            "total": self.total.public(),
            "per_target": {k: v.public() for k, v in self.per_target.items()},
        }

    # -- costing -----------------------------------------------------------
    def _count(self, texts, model_id: str) -> int:
        from octt import generation

        return generation.count_text_tokens(texts, model_id, offline=not self.execute)

    def estimate_usd(self, keys, conv_id: str, message: str) -> float:
        """What one turn across *keys* would cost, given each target's history."""
        total = 0.0
        for key in keys:
            t = self._targets.get(key)
            if t is None:
                continue
            history = self._hist.get(conv_id, {}).get(key, [])
            texts = [m["content"] for m in history] + [message]
            ptok = self._count(texts, t.model_id)
            total += ptok * t.price_prefill / 1e6
            total += self.max_tokens * t.price_sample / 1e6
        return total

    def remaining_usd(self) -> float:
        return max(0.0, self.budget_usd - self.total.usd)

    # -- sampling ----------------------------------------------------------
    def _sampler(self, target: ChatTarget):
        from octt import generation, tinker_client

        if self._runtime is None:
            model_ids = sorted({t.model_id for t in self._targets.values()})
            self._runtime = tinker_client.create_runtime(
                model_ids,
                tinker_client.TinkerClientConfig(dry_run=not self.execute),
            )
        if target.key not in self._samplers:
            self._samplers[target.key] = generation.make_sampler(
                self._runtime,
                target.model_id,
                model_path=target.model_path,
                tag=f"chat:{target.key}",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                thinking=False,  # eval parity: direct-answer renderer
            )
        return self._samplers[target.key]

    def reset(self, conv_id: str) -> None:
        with self._lock:
            self._hist.pop(conv_id, None)

    def history(self, conv_id: str, key: str) -> list:
        return list(self._hist.get(conv_id, {}).get(key, []))

    def send(self, conv_id: str, message: str, keys) -> dict:
        """Append *message* to each selected target's conversation and sample a reply."""
        keys = [k for k in keys if k in self._targets]
        if not keys:
            raise ValueError("no valid targets selected")

        estimate = self.estimate_usd(keys, conv_id, message)
        with self._lock:
            # The browser fires one request per model so replies land independently,
            # so several turns can be in flight at once. Reserve this turn's estimate
            # under the lock or concurrent checks all read the same spent total and
            # collectively overshoot the budget.
            if self.execute and self.total.usd + self._reserved + estimate > self.budget_usd:
                free = max(0.0, self.budget_usd - self.total.usd - self._reserved)
                raise BudgetExceeded(
                    f"this turn would cost about ${estimate:.4f}; ${free:.4f} of the "
                    f"${self.budget_usd:.2f} budget is uncommitted"
                )
            self._reserved += estimate
            convo = self._hist.setdefault(conv_id, {})
            for key in keys:
                convo.setdefault(key, []).append({"role": "user", "content": message})
            pending = {key: list(convo[key]) for key in keys}

        try:
            replies = self._sample_all(keys, pending)
        except BaseException:
            with self._lock:
                self._reserved = max(0.0, self._reserved - estimate)
            raise

        out = {}
        with self._lock:
            self._reserved = max(0.0, self._reserved - estimate)
            convo = self._hist.setdefault(conv_id, {})
            for key in keys:
                t = self._targets[key]
                text = replies[key]
                convo.setdefault(key, []).append({"role": "assistant", "content": text})
                ptok = self._count([m["content"] for m in pending[key]], t.model_id)
                stok = self._count([text], t.model_id)
                usd = ptok * t.price_prefill / 1e6 + stok * t.price_sample / 1e6
                self.per_target[key].add(ptok, stok, usd)
                self.total.add(ptok, stok, usd)
                out[key] = {
                    "text": text,
                    "prompt_tokens": ptok,
                    "sample_tokens": stok,
                    # Unrounded: a single reply costs on the order of 1e-5 USD, and
                    # rounding here would discard a few percent of it. Formatting is
                    # the browser's job.
                    "usd": usd,
                    "stub": not self.execute,
                }
        return {
            "replies": out,
            "total": self.total.public(),
            "per_target": {k: v.public() for k, v in self.per_target.items()},
            "remaining_usd": round(self.remaining_usd(), 6),
            "estimate_usd": round(estimate, 6),
        }

    def _sample_all(self, keys, pending) -> dict:
        from octt import generation

        async def run_all():
            tasks = [
                generation.complete_async(self._sampler(self._targets[k]), pending[k])
                for k in keys
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(run_all())
        out = {}
        for key, res in zip(keys, results):
            if isinstance(res, BaseException):
                out[key] = f"[sampling failed: {type(res).__name__}: {res}]"
            else:
                out[key] = res
        return out
