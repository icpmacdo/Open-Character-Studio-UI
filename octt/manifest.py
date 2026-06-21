"""Run manifests, checkpoint registry, and content-hash caching.

Training on Tinker is billed per token and produces ``tinker://`` checkpoint
handles whose weights live server-side. The handle is the only thing that lets
us reuse an expensive artifact, so losing it means re-paying for it. This module
is the bookkeeping layer that makes a sweep *idempotent and resumable*, per the
rules in ``docs/COST_CONTROLS.md``:

  - Persist every checkpoint URI the instant it is created, atomically.
  - Track *both* checkpoint kinds at each stage boundary: ``state`` (optimizer
    state, for resuming/continuing training) and ``sampler`` (weights, for
    inference / generating the next stage's data).
  - Give every (model, persona, stage) a deterministic ``run_id`` so a crashed
    sweep skips finished stages instead of re-sampling the teacher.
  - Cache generated data and judge verdicts by content hash so a retried
    *training* step never re-hits the *sampling* APIs.

Nothing here imports ``tinker``; it is pure-Python bookkeeping that the dry-run
tier exercises in full.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

MANIFEST_BASE_NAME = "manifest.json"
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Stable hashing
# ---------------------------------------------------------------------------


def _canonical(obj: Any) -> Any:
    """Return a JSON-stable representation of *obj* for hashing.

    Dataclasses become sorted dicts; mappings are key-sorted; sequences keep
    order. The result is deterministic across processes (no ``hash()`` salt).
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, Mapping):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float):
        # Avoid "1.0" vs "1" drift between equal values.
        return format(obj, ".12g")
    return obj


def stable_hash(*parts: Any, length: int = 16) -> str:
    """Deterministic short hex digest of any JSON-able / dataclass parts."""
    payload = json.dumps([_canonical(p) for p in parts], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def config_hash(config: Any, length: int = 12) -> str:
    """Stable hash of a (possibly nested) recipe config dataclass."""
    return stable_hash(config, length=length)


def content_hash(*parts: Any, length: int = 16) -> str:
    """Stable hash for data/verdict caching keyed on its inputs.

    Use for keys like ``(persona, teacher, prompt_set, sampling_params)`` so a
    retried run reads cached artifacts instead of re-sampling.
    """
    return stable_hash(*parts, length=length)


def run_id(model: str, persona: str, config: Any, length: int = 12) -> str:
    """Deterministic id for one (model, persona) recipe run.

    Folds the config hash in so that changing a hyperparameter starts a fresh,
    non-colliding run directory rather than resuming an incompatible one.
    """
    safe_model = model.replace("/", "-")
    return f"{persona}-{safe_model}-{config_hash(config, length=length)}"


# ---------------------------------------------------------------------------
# Atomic JSON I/O
# ---------------------------------------------------------------------------


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON to *path* atomically (temp file in same dir, then rename).

    A rename is atomic on POSIX, so a crash mid-write can never leave a
    truncated manifest that would strand a checkpoint URI.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Checkpoint records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageCheckpoint:
    """The output of a training/merge stage: both Tinker checkpoint kinds.

    ``state_path`` resumes/continues training; ``sampler_path`` generates the
    next stage's data and runs eval. Saving only one silently breaks the next
    stage, so both are first-class (``sampler`` may legitimately be ``None`` for
    a resume-only rolling save, but stage boundaries should carry both).
    """

    sampler_path: str | None = None
    state_path: str | None = None
    local_path: str | None = None  # local adapter dir (e.g. a linear-merged adapter)
    step: int | None = None
    config_hash: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """A usable stage output exposes a samplable handle or a local artifact."""
        return bool(self.sampler_path or self.local_path)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.sampler_path is not None:
            d["sampler_path"] = self.sampler_path
        if self.state_path is not None:
            d["state_path"] = self.state_path
        if self.local_path is not None:
            d["local_path"] = self.local_path
        if self.step is not None:
            d["step"] = self.step
        if self.config_hash is not None:
            d["config_hash"] = self.config_hash
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StageCheckpoint":
        return cls(
            sampler_path=d.get("sampler_path"),
            state_path=d.get("state_path"),
            local_path=d.get("local_path"),
            step=d.get("step"),
            config_hash=d.get("config_hash"),
            extra=dict(d.get("extra", {})),
        )

    @property
    def is_dry_run(self) -> bool:
        return bool(self.sampler_path and self.sampler_path.startswith("tinker://dry-run/"))


def dry_run_checkpoint(stage: str, *key: Any, step: int | None = None) -> StageCheckpoint:
    """Mint a deterministic placeholder checkpoint for the dry-run tier.

    The ``tinker://dry-run/...`` scheme makes it obvious in manifests that no
    real weights were saved, while still flowing through every code path
    (recording, skip-if-exists, round-trip verify) exactly like a real handle.
    """
    h = stable_hash(stage, *key)
    return StageCheckpoint(
        sampler_path=f"tinker://dry-run/{stage}-{h}/sampler_weights/final",
        state_path=f"tinker://dry-run/{stage}-{h}/state/final",
        step=step,
        config_hash=h,
    )


@dataclass
class RunManifest:
    """Append-only record of one run's stage checkpoints, persisted atomically.

    Layout on disk: ``<out_dir>/manifest.json``. Each recorded stage stores its
    ``StageCheckpoint`` plus a timestamp, so a resumed sweep can look up a
    finished stage and skip it (see :meth:`stage` / :meth:`has_stage`).
    """

    run_id: str
    model: str
    persona: str
    out_dir: Path
    config_hash: str | None = None
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    schema_version: int = SCHEMA_VERSION

    @property
    def path(self) -> Path:
        return Path(self.out_dir) / MANIFEST_BASE_NAME

    # -- persistence --------------------------------------------------------

    @classmethod
    def load_or_create(
        cls,
        out_dir: Path,
        *,
        model: str,
        persona: str,
        config: Any | None = None,
    ) -> "RunManifest":
        """Load an existing manifest from *out_dir* or create a fresh one.

        If a manifest exists but was written for a different config hash, it is
        kept (the deterministic ``run_id`` already namespaces by config); the
        loaded record wins so finished stages resume.
        """
        out_dir = Path(out_dir)
        cfg_hash = config_hash(config) if config is not None else None
        existing = out_dir / MANIFEST_BASE_NAME
        if existing.exists():
            data = json.loads(existing.read_text())
            return cls(
                run_id=data["run_id"],
                model=data.get("model", model),
                persona=data.get("persona", persona),
                out_dir=out_dir,
                config_hash=data.get("config_hash", cfg_hash),
                stages=data.get("stages", {}),
                created_at=data.get("created_at", time.time()),
                updated_at=data.get("updated_at", time.time()),
                schema_version=data.get("schema_version", SCHEMA_VERSION),
            )
        rid = run_id(model, persona, config) if config is not None else f"{persona}-{model}"
        manifest = cls(
            run_id=rid,
            model=model,
            persona=persona,
            out_dir=out_dir,
            config_hash=cfg_hash,
        )
        manifest.flush()
        return manifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "model": self.model,
            "persona": self.persona,
            "config_hash": self.config_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stages": self.stages,
        }

    def flush(self) -> None:
        self.updated_at = time.time()
        atomic_write_json(self.path, self.to_dict())

    # -- stage records ------------------------------------------------------

    def record_stage(
        self,
        stage: str,
        checkpoint: StageCheckpoint,
        **extra: Any,
    ) -> StageCheckpoint:
        """Persist a stage's checkpoint to the manifest atomically and return it."""
        record = checkpoint.to_dict()
        record["recorded_at"] = time.time()
        if extra:
            record.setdefault("extra", {}).update(extra)
        self.stages[stage] = record
        self.flush()
        return checkpoint

    def stage(self, stage: str) -> StageCheckpoint | None:
        record = self.stages.get(stage)
        if record is None:
            return None
        return StageCheckpoint.from_dict(record)

    def has_stage(self, stage: str) -> bool:
        """True if *stage* already produced a usable (sampler) checkpoint."""
        ckpt = self.stage(stage)
        return ckpt is not None and ckpt.ok
