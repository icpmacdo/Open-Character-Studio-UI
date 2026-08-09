"""On-policy distillation (OPD) with an ASYMMETRIC teacher context.

The stock on-policy distillation recipe
(``tinker_cookbook/distillation/train_on_policy.py``) scores teacher and student
on *the same* rendered sequence. Character acquisition by OPD needs the teacher
to see the constitution and the student not to::

    student: [user prompt][same sampled completion]
    teacher: [system constitution][user prompt][same sampled completion]

The teacher's extra system block shifts every token index, so "the same
completion" is only meaningful if the completion's *token ids* are provably
identical in both sequences and their positions are located exactly. A silent
off-by-one here does not crash: it trains the student against teacher
log-probabilities for the wrong tokens, and the run looks fine.

This module is that alignment layer, written first-party because
``tinker-cookbook/`` is vendored read-only (see
``docs/IMPLEMENTATION_READINESS_2026-07-27.md``, "Stock-recipe gaps to
isolate"). It:

1. samples the completion from the student (:func:`sample_group`);
2. renders both prefixes with the SAME model-family renderer
   (:class:`AsymmetricRenderer` refuses two different renderers);
3. appends the identical completion text to both views;
4. tokenizes both full sequences through that renderer;
5. locates the completion span in each and asserts the two spans are
   token-for-token identical (:func:`align_pair`);
6. exposes the teacher slice so log-probabilities are read only over aligned
   completion tokens (:meth:`AlignedPair.completion_logprobs`);
7. raises :class:`AlignmentRefusal` — refusing the whole batch — on ANY
   ambiguity. There is no "best effort" path.

**KL telemetry.** The cookbook's ``kl_policy_base`` is a *signed k1*
log-probability difference. It is the right quantity for the training objective
(the distillation advantage is the signed teacher-minus-student signal) and the
wrong one for monitoring, because it is negative as often as not and does not
estimate a KL. Both are computed here under names that cannot be confused:
:func:`teacher_minus_student` (signed, drives training) and :func:`kl_k3`
(nonnegative k3, monitoring only). Nothing in the vendored tree is patched, so
``kl_policy_base`` keeps its meaning wherever it is already banked.

Heavy dependencies (``tinker``, ``torch``, ``tinker_cookbook``) are imported
lazily inside functions: importing this module and running its unit tests must
work with no training stack installed.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import generation, manifest, models
from .constitution import Constitution, character_system_prompt
from .tinker_client import TinkerRuntime

logger = logging.getLogger(__name__)

Message = dict[str, Any]

# Bumped whenever the alignment CONTRACT changes (what counts as an aligned
# pair), not when unrelated code moves. Stamped into every aligned pair, smoke
# report, and metrics row so banked OPD numbers are attributable to a contract.
RENDER_CONTRACT_VERSION = "opd-asymmetric-v1"

# Fingerprint probe for the tokenizer half of the render contract: multi-byte
# characters, combining marks, an emoji outside the BMP, and runs of mixed
# whitespace. Two tokenizers that agree here agree on the boundaries this layer
# depends on; one that does not is a different instrument.
TOKENIZER_PROBE = "\n Hello, wörld — 世界 🙂\ttabs  and   spaces ́\n"

# Envelope for one full teacher sequence: constitution system prompt + user
# prompt + a max-length response, with headroom. A sequence over this is
# REFUSED rather than truncated — a right-truncated teacher sequence loses the
# tail of the completion it is supposed to be scoring.
DEFAULT_MAX_SEQUENCE_TOKENS = 8192

# exp() overflows around logr = 709. A token whose reference/policy
# log-probability gap exceeds this is already a saturated KL signal, so the
# estimator clamps and *counts* the clamps (reported as `..._clamped_tokens`)
# rather than raising or silently returning inf.
K3_LOGR_CLAMP = 60.0

# Same scrub as octt.generation: renderer control tokens must not survive into
# the completion text that gets re-rendered into both views.
_RENDERER_TOKEN = re.compile(r"<\|[a-zA-Z0-9_]{1,64}\|>")


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------

REASON_EMPTY_COMPLETION = "empty_completion"
REASON_PREFIX_NOT_A_PREFIX = "prefix_not_a_prefix"
REASON_COMPLETION_MISMATCH = "completion_token_mismatch"
REASON_SAMPLED_TOKEN_MISMATCH = "sampled_token_mismatch"
REASON_LOSS_MASK_MISMATCH = "loss_mask_mismatch"
REASON_DECODE_MISMATCH = "completion_decode_mismatch"
REASON_TRUNCATED = "sequence_over_max_tokens"
REASON_RENDERER_MISMATCH = "renderer_contract_mismatch"
REASON_CONSTITUTION_ABSENT = "constitution_absent_from_teacher_context"
REASON_CONTROL_TOKEN_IN_COMPLETION = "control_token_in_completion"
REASON_BATCH_SHAPE = "batch_shape_mismatch"


class AlignmentRefusal(RuntimeError):
    """The batch could not be proven aligned, so it is refused.

    Every construction path in this module raises this rather than returning a
    partially-aligned result. Callers must not catch it to skip a sample: a
    single ambiguous sample means the renderer/tokenizer contract does not hold
    for this batch, and the samples that "looked fine" were checked by the same
    logic that just failed.
    """

    def __init__(self, reason: str, detail: str, *, index: int | None = None) -> None:
        self.reason = reason
        self.detail = detail
        self.index = index
        where = "" if index is None else f" (sample {index})"
        super().__init__(f"OPD alignment refused [{reason}]{where}: {detail}")


class SmokeGateError(RuntimeError):
    """A paid training request was attempted without a passing single-response smoke."""


# ---------------------------------------------------------------------------
# Pilot configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OPDConfig:
    """The OPD pilot (readiness doc, "Starting OPD pilot").

    Defaults ARE the pilot table; they are not tuning knobs to be edited in
    place. Change them by constructing a new config at the call site so the
    config hash records the divergence.
    """

    student_model: str = "Qwen/Qwen3.5-4B"
    teacher_model: str = "Qwen/Qwen3.5-4B"
    lora_rank: int = 32
    learning_rate: float = 1e-4
    prompts_per_batch: int = 8
    samples_per_prompt: int = 4
    temperature: float = 1.0
    max_response_tokens: int = 512
    # "Distillation KL coefficient 1": the multiplier on the signed
    # teacher-minus-student per-token signal that becomes the advantage.
    kl_coefficient: float = 1.0
    max_steps: int = 20
    save_every: int = 5
    eval_every: int = 5
    # Not part of the pilot table: the refusal threshold for a full rendered
    # sequence (see DEFAULT_MAX_SEQUENCE_TOKENS).
    max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS
    # Teacher weights. None = the base model conditioned on the constitution,
    # which is the OPD arm's definition of "teacher".
    teacher_checkpoint: str | None = None
    # Frozen reference for base-relative k3 (comparison with the DPO and RL
    # arms). None = the unmodified base student.
    base_reference_checkpoint: str | None = None

    def __post_init__(self) -> None:
        for name in ("prompts_per_batch", "samples_per_prompt", "max_steps", "lora_rank"):
            if getattr(self, name) < 1:
                raise ValueError(f"OPDConfig.{name} must be >= 1")
        if self.max_response_tokens < 1:
            raise ValueError("OPDConfig.max_response_tokens must be >= 1")
        if self.temperature <= 0:
            raise ValueError("OPDConfig.temperature must be > 0 (OPD samples on-policy)")

    @property
    def samples_per_step(self) -> int:
        return self.prompts_per_batch * self.samples_per_prompt


OPD_PILOT = OPDConfig()


# ---------------------------------------------------------------------------
# Render contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderContract:
    """Identity of the renderer+tokenizer that produced an aligned pair.

    Two views may only be compared when they were rendered by the same
    contract. Stamped into aligned pairs, smoke reports, and metrics rows; a
    contract change invalidates comparability of banked OPD numbers exactly the
    way a judge-instrument change does elsewhere in this project.
    """

    renderer_name: str
    renderer_class: str
    tokenizer_fingerprint: str
    contract_version: str = RENDER_CONTRACT_VERSION

    @property
    def instrument_id(self) -> str:
        return manifest.stable_hash(
            self.contract_version,
            self.renderer_name,
            self.renderer_class,
            self.tokenizer_fingerprint,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "contract_version": self.contract_version,
            "renderer_name": self.renderer_name,
            "renderer_class": self.renderer_class,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "instrument_id": self.instrument_id,
        }


def _to_ints(model_input: Any) -> tuple[int, ...]:
    """Token ids from a ``tinker.ModelInput`` (or a plain sequence in tests)."""
    if hasattr(model_input, "to_ints"):
        return tuple(int(t) for t in model_input.to_ints())
    return tuple(int(t) for t in model_input)


def _to_weights(weights: Any) -> tuple[float, ...]:
    """Per-token weights from a torch tensor, numpy array, or plain sequence."""
    if hasattr(weights, "tolist"):
        weights = weights.tolist()
    return tuple(float(w) for w in weights)


@dataclass(frozen=True)
class RendererHandle:
    """One model-family renderer + its tokenizer, used for one view."""

    model_id: str
    renderer_name: str
    renderer: Any
    tokenizer: Any
    # The renderer's LAST_ASSISTANT_MESSAGE enum, resolved lazily on the real
    # path. None falls back to the renderer's own default (which is
    # last-assistant for every cookbook renderer, and irrelevant here anyway:
    # these conversations carry exactly one assistant message).
    train_on_what: Any = None

    @classmethod
    def from_runtime(cls, runtime: TinkerRuntime, model_id: str) -> RendererHandle:
        binding = runtime.renderer_binding(model_id)
        train_on_what = None
        try:  # pragma: no cover - needs the vendored cookbook importable
            from tinker_cookbook.renderers import TrainOnWhat

            train_on_what = TrainOnWhat.LAST_ASSISTANT_MESSAGE
        except ImportError:
            logger.warning("TrainOnWhat unavailable; using the renderer's default weighting")
        return cls(
            model_id=model_id,
            renderer_name=binding.renderer_name,
            renderer=binding.renderer,
            tokenizer=binding.tokenizer,
            train_on_what=train_on_what,
        )

    def contract(self) -> RenderContract:
        encoded = self.tokenizer.encode(TOKENIZER_PROBE, add_special_tokens=False)
        probe = tuple(int(t) for t in encoded)
        # The probe only SAMPLES the tokenizer's behavior: two vocabularies can
        # agree on it and disagree everywhere else. Identifying attributes are
        # folded in so a swapped tokenizer changes the fingerprint even when the
        # probe cannot see the difference.
        identity = tuple(
            (attr, str(value))
            for attr in ("name_or_path", "vocab_size")
            if (value := getattr(self.tokenizer, attr, None)) is not None
        )
        return RenderContract(
            renderer_name=self.renderer_name,
            renderer_class=type(self.renderer).__name__,
            tokenizer_fingerprint=manifest.stable_hash(TOKENIZER_PROBE, probe, identity),
        )

    def prefix_tokens(self, messages: Sequence[Message]) -> tuple[int, ...]:
        """Tokens of the generation prompt — everything before the completion."""
        return _to_ints(self.renderer.build_generation_prompt(list(messages)))

    def full_render(self, messages: Sequence[Message]) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Tokens + per-token training weights for a conversation ending in the
        assistant message. This is the renderer's OWN tokenization of the whole
        sequence, not a concatenation, so it is what catches merges across the
        prefix/completion boundary."""
        if self.train_on_what is None:
            rendered = self.renderer.build_supervised_example(list(messages))
        else:
            rendered = self.renderer.build_supervised_example(
                list(messages), train_on_what=self.train_on_what
            )
        model_input, weights = rendered
        return _to_ints(model_input), _to_weights(weights)

    def decode(self, tokens: Sequence[int]) -> str:
        return str(self.tokenizer.decode(list(tokens)))


@dataclass(frozen=True)
class AsymmetricRenderer:
    """The student and teacher views, pinned to one render contract.

    Constructing this with two different contracts is itself a refusal: the
    whole point of the layer is that both views are produced by the same
    model-family renderer, so a renderer-version drift between them must never
    reach the alignment step.
    """

    student: RendererHandle
    teacher: RendererHandle

    def __post_init__(self) -> None:
        student_contract = self.student.contract()
        teacher_contract = self.teacher.contract()
        if student_contract != teacher_contract:
            raise AlignmentRefusal(
                REASON_RENDERER_MISMATCH,
                f"student view renders with {student_contract.to_dict()} but teacher view "
                f"renders with {teacher_contract.to_dict()}; both views must use the same "
                "model-family renderer and tokenizer",
            )

    @classmethod
    def from_runtime(
        cls, runtime: TinkerRuntime, student_model: str, teacher_model: str | None = None
    ) -> AsymmetricRenderer:
        student = RendererHandle.from_runtime(runtime, student_model)
        teacher = (
            student
            if teacher_model in (None, student_model)
            else RendererHandle.from_runtime(runtime, teacher_model)
        )
        return cls(student=student, teacher=teacher)

    def contract(self) -> RenderContract:
        return self.student.contract()


# ---------------------------------------------------------------------------
# Prompt mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptPair:
    """One prompt in both views: student (bare) and teacher (constitutioned)."""

    prompt: str
    system: str
    student_messages: tuple[Message, ...]
    teacher_messages: tuple[Message, ...]


def prompt_pair(prompt: str, system: str) -> PromptPair:
    """Map one user prompt to the student and teacher views.

    The student view carries the user turn only; the teacher view prepends the
    constitution as a system message. Nothing else differs — any other delta
    would make the teacher's log-probabilities a measurement of that delta
    rather than of the character.
    """
    student: tuple[Message, ...] = ({"role": "user", "content": prompt},)
    teacher: tuple[Message, ...] = (
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    )
    return PromptPair(
        prompt=prompt, system=system, student_messages=student, teacher_messages=teacher
    )


def constitution_prompt_pairs(
    constitution: Constitution, prompts: Sequence[str], *, student_model: str
) -> list[PromptPair]:
    """Prompt pairs whose teacher system block is the paper's character prompt."""
    system = character_system_prompt(constitution, models.assistant_name(student_model))
    return [prompt_pair(p, system) for p in prompts]


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignedPair:
    """A student/teacher sequence pair with a proven-identical completion span."""

    prompt: str
    completion: str
    student_tokens: tuple[int, ...]
    teacher_tokens: tuple[int, ...]
    student_prefix_len: int
    teacher_prefix_len: int
    completion_tokens: tuple[int, ...]
    contract: RenderContract

    @property
    def num_completion_tokens(self) -> int:
        return len(self.completion_tokens)

    @property
    def constitution_tokens(self) -> int:
        """How many tokens of extra context the teacher sees."""
        return self.teacher_prefix_len - self.student_prefix_len

    def tokens(self, side: str) -> tuple[int, ...]:
        return self.student_tokens if _side(side) == "student" else self.teacher_tokens

    def prefix_len(self, side: str) -> int:
        return self.student_prefix_len if _side(side) == "student" else self.teacher_prefix_len

    def completion_logprobs(self, logprobs: Sequence[float], *, side: str) -> list[float]:
        """Slice a full-sequence log-probability array down to the completion.

        ``compute_logprobs`` returns one entry per token of the sequence it was
        given, entry ``i`` being the log-probability of token ``i`` in context.
        The completion occupies ``[prefix_len, len(tokens))`` in that array —
        different indices on each side, which is the whole reason this layer
        exists. Length is validated, never assumed.
        """
        side = _side(side)
        expected = len(self.tokens(side))
        if len(logprobs) != expected:
            raise AlignmentRefusal(
                REASON_BATCH_SHAPE,
                f"{side} log-probability array has {len(logprobs)} entries for a "
                f"{expected}-token sequence",
            )
        sliced = [float(x) for x in logprobs[self.prefix_len(side) :]]
        if len(sliced) != self.num_completion_tokens:
            raise AlignmentRefusal(
                REASON_BATCH_SHAPE,
                f"{side} completion slice has {len(sliced)} entries, expected "
                f"{self.num_completion_tokens}",
            )
        return sliced

    def target_mask(self, side: str = "student") -> list[float]:
        """Loss mask over left-shifted targets (``tokens[1:]``).

        Entry ``j`` scores the prediction of ``tokens[j + 1]``, so the
        completion span ``[P, L)`` masks in at ``[P - 1, L - 1)``.
        """
        side = _side(side)
        total = len(self.tokens(side))
        prefix = self.prefix_len(side)
        return [1.0 if j + 1 >= prefix else 0.0 for j in range(total - 1)]

    def scatter_over_targets(
        self, values: Sequence[float], *, side: str = "student"
    ) -> list[float]:
        """Place per-completion-token *values* on the left-shifted target axis."""
        side = _side(side)
        if len(values) != self.num_completion_tokens:
            raise AlignmentRefusal(
                REASON_BATCH_SHAPE,
                f"expected {self.num_completion_tokens} completion values, got {len(values)}",
            )
        total = len(self.tokens(side))
        prefix = self.prefix_len(side)
        out = [0.0] * (total - 1)
        for offset, value in enumerate(values):
            out[prefix - 1 + offset] = float(value)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "completion_chars": len(self.completion),
            "student_sequence_tokens": len(self.student_tokens),
            "teacher_sequence_tokens": len(self.teacher_tokens),
            "student_prefix_tokens": self.student_prefix_len,
            "teacher_prefix_tokens": self.teacher_prefix_len,
            "completion_tokens": self.num_completion_tokens,
            "constitution_tokens": self.constitution_tokens,
            "contract": self.contract.to_dict(),
        }


def _side(side: str) -> str:
    if side not in ("student", "teacher"):
        raise ValueError(f"side must be 'student' or 'teacher', got {side!r}")
    return side


def _check_prefix(
    full: tuple[int, ...], prefix: tuple[int, ...], side: str, index: int | None
) -> None:
    """The generation prompt must be a literal prefix of the full render.

    When it is not, the renderer rewrites history as the conversation grows
    (thinking-block stripping, injected defaults, effort directives). The
    completion span cannot be located by subtraction in that case, so the pair
    is refused rather than guessed at.
    """
    if len(prefix) >= len(full):
        raise AlignmentRefusal(
            REASON_PREFIX_NOT_A_PREFIX,
            f"{side} generation prompt is {len(prefix)} tokens but its full render is "
            f"{len(full)} — no completion tokens were located",
            index=index,
        )
    if full[: len(prefix)] != prefix:
        first = next(
            (i for i in range(len(prefix)) if full[i] != prefix[i]),
            len(prefix),
        )
        raise AlignmentRefusal(
            REASON_PREFIX_NOT_A_PREFIX,
            f"{side} generation prompt is not a prefix of the full render "
            f"(first divergence at token {first}); this renderer rewrites earlier "
            "tokens when the assistant message is appended",
            index=index,
        )


def _check_mask(
    weights: tuple[float, ...], prefix_len: int, total: int, side: str, index: int | None
) -> None:
    """The renderer's own loss weights must select exactly the completion span."""
    if len(weights) != total:
        raise AlignmentRefusal(
            REASON_LOSS_MASK_MISMATCH,
            f"{side} render returned {len(weights)} weights for {total} tokens",
            index=index,
        )
    if any(w != 0 for w in weights[:prefix_len]):
        raise AlignmentRefusal(
            REASON_LOSS_MASK_MISMATCH,
            f"{side} render puts training weight on prompt tokens; the completion span "
            "cannot be identified with the trained span",
            index=index,
        )
    if any(w <= 0 for w in weights[prefix_len:]):
        raise AlignmentRefusal(
            REASON_LOSS_MASK_MISMATCH,
            f"{side} render leaves {sum(1 for w in weights[prefix_len:] if w <= 0)} "
            "completion tokens unweighted",
            index=index,
        )


def _normalize_completion_text(text: str) -> str:
    """Control-token-free, whitespace-collapsed view for the decode round-trip."""
    return " ".join(_RENDERER_TOKEN.sub("", text).split())


def align_pair(
    renderer: AsymmetricRenderer,
    pair: PromptPair,
    completion: str,
    *,
    sampled_tokens: Sequence[int] | None = None,
    max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS,
    expected_contract: RenderContract | None = None,
    index: int | None = None,
) -> AlignedPair:
    """Render both views around one completion and prove the spans align.

    Returns an :class:`AlignedPair` or raises :class:`AlignmentRefusal`. There
    is no third outcome.

    ``sampled_tokens`` (the ids the policy actually emitted) is checked against
    the renderer's canonical re-tokenization when supplied. They must match
    exactly: the training step reuses the sampler's per-token log-probabilities
    as importance weights, and those are only valid for the tokens that were
    sampled.
    """
    contract = renderer.contract()
    if expected_contract is not None and contract != expected_contract:
        raise AlignmentRefusal(
            REASON_RENDERER_MISMATCH,
            f"renderer contract is {contract.instrument_id} but this batch was gated "
            f"against {expected_contract.instrument_id}; the renderer or tokenizer "
            "changed underneath the run",
            index=index,
        )

    if not completion or not completion.strip():
        raise AlignmentRefusal(
            REASON_EMPTY_COMPLETION,
            "the sampled completion is empty or whitespace-only; there is nothing to "
            "align and nothing to distill",
            index=index,
        )

    control = _RENDERER_TOKEN.search(completion)
    if control is not None:
        raise AlignmentRefusal(
            REASON_CONTROL_TOKEN_IN_COMPLETION,
            f"the completion text contains the renderer control token {control.group(0)!r}; "
            "re-rendering it turns a piece of the model's prose into a structural token, so "
            "the span would not be the sampled completion (octt.generation scrubs these — a "
            "surviving one means the sampling path changed)",
            index=index,
        )

    assistant: Message = {"role": "assistant", "content": completion}

    student_prefix = renderer.student.prefix_tokens(pair.student_messages)
    student_full, student_weights = renderer.student.full_render(
        [*pair.student_messages, assistant]
    )
    _check_prefix(student_full, student_prefix, "student", index)
    _check_mask(student_weights, len(student_prefix), len(student_full), "student", index)
    student_completion = student_full[len(student_prefix) :]

    teacher_prefix = renderer.teacher.prefix_tokens(pair.teacher_messages)
    teacher_full, teacher_weights = renderer.teacher.full_render(
        [*pair.teacher_messages, assistant]
    )
    _check_prefix(teacher_full, teacher_prefix, "teacher", index)
    _check_mask(teacher_weights, len(teacher_prefix), len(teacher_full), "teacher", index)
    teacher_completion = teacher_full[len(teacher_prefix) :]

    if student_completion != teacher_completion:
        raise AlignmentRefusal(
            REASON_COMPLETION_MISMATCH,
            f"completion tokenizes to {len(student_completion)} tokens after the student "
            f"prefix but {len(teacher_completion)} after the teacher prefix "
            f"(student={_preview_tokens(student_completion)}, "
            f"teacher={_preview_tokens(teacher_completion)}); the teacher's extra context "
            "changed the tokenization of the shared completion",
            index=index,
        )

    if sampled_tokens is not None:
        sampled = tuple(int(t) for t in sampled_tokens)
        if sampled != student_completion:
            raise AlignmentRefusal(
                REASON_SAMPLED_TOKEN_MISMATCH,
                f"the policy emitted {len(sampled)} tokens but re-rendering the decoded "
                f"completion yields {len(student_completion)} "
                f"(sampled={_preview_tokens(sampled)}, "
                f"rendered={_preview_tokens(student_completion)}); the sampler's "
                "log-probabilities do not describe the tokens that would be trained",
                index=index,
            )

    decoded = renderer.student.decode(student_completion)
    if _normalize_completion_text(decoded) != _normalize_completion_text(completion):
        raise AlignmentRefusal(
            REASON_DECODE_MISMATCH,
            f"completion tokens decode to {_normalize_completion_text(decoded)[:120]!r} but "
            f"the completion text is {_normalize_completion_text(completion)[:120]!r}; the "
            "located span is not the completion",
            index=index,
        )

    head = pair.teacher_messages[0] if pair.teacher_messages else {}
    conditioned = head.get("role") == "system" and str(head.get("content", "")).strip()
    if conditioned and len(teacher_prefix) <= len(student_prefix):
        raise AlignmentRefusal(
            REASON_CONSTITUTION_ABSENT,
            f"the teacher prefix ({len(teacher_prefix)} tokens) is no longer than the "
            f"student prefix ({len(student_prefix)} tokens); this renderer dropped the "
            "system block, so the teacher is not constitution-conditioned",
            index=index,
        )

    for side, tokens in (("student", student_full), ("teacher", teacher_full)):
        if len(tokens) > max_sequence_tokens:
            raise AlignmentRefusal(
                REASON_TRUNCATED,
                f"{side} sequence is {len(tokens)} tokens, over the {max_sequence_tokens} "
                "limit; truncating it would silently drop the tail of the completion "
                "being scored",
                index=index,
            )

    return AlignedPair(
        prompt=pair.prompt,
        completion=completion,
        student_tokens=student_full,
        teacher_tokens=teacher_full,
        student_prefix_len=len(student_prefix),
        teacher_prefix_len=len(teacher_prefix),
        completion_tokens=student_completion,
        contract=contract,
    )


def _preview_tokens(tokens: Sequence[int], n: int = 8) -> str:
    head = list(tokens[:n])
    return f"{head}{'...' if len(tokens) > n else ''}"


@dataclass(frozen=True)
class AlignedBatch:
    """A whole batch of aligned pairs under one contract, or nothing."""

    pairs: tuple[AlignedPair, ...]
    contract: RenderContract

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def num_completion_tokens(self) -> int:
        return sum(p.num_completion_tokens for p in self.pairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_pairs": len(self.pairs),
            "num_completion_tokens": self.num_completion_tokens,
            "contract": self.contract.to_dict(),
        }


def align_batch(
    renderer: AsymmetricRenderer,
    pairs: Sequence[PromptPair],
    completions: Sequence[str],
    *,
    sampled_tokens: Sequence[Sequence[int]] | None = None,
    max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS,
    expected_contract: RenderContract | None = None,
) -> AlignedBatch:
    """Align a whole batch, refusing ALL of it if any sample is ambiguous.

    Dropping the bad sample and training on the rest would be the wrong repair:
    the samples that passed were checked by the logic that just failed, so the
    batch's alignment is unproven as a whole.
    """
    if len(pairs) != len(completions):
        raise AlignmentRefusal(
            REASON_BATCH_SHAPE, f"{len(pairs)} prompt pairs but {len(completions)} completions"
        )
    if sampled_tokens is not None and len(sampled_tokens) != len(pairs):
        raise AlignmentRefusal(
            REASON_BATCH_SHAPE,
            f"{len(pairs)} prompt pairs but {len(sampled_tokens)} sampled token sequences",
        )
    if not pairs:
        raise AlignmentRefusal(REASON_BATCH_SHAPE, "empty batch")

    contract = expected_contract or renderer.contract()
    aligned = tuple(
        align_pair(
            renderer,
            pair,
            completion,
            sampled_tokens=None if sampled_tokens is None else sampled_tokens[i],
            max_sequence_tokens=max_sequence_tokens,
            expected_contract=contract,
            index=i,
        )
        for i, (pair, completion) in enumerate(zip(pairs, completions))
    )
    return AlignedBatch(pairs=aligned, contract=contract)


# ---------------------------------------------------------------------------
# KL telemetry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KLTelemetry:
    """Nonnegative k3 KL of a policy against a reference, in nats.

    ``label`` names the reference ("teacher", "base_ref") and prefixes every
    metric key, so a teacher-relative number can never be read as a
    base-relative one.
    """

    label: str
    mean_token_kl_nats: float
    mean_response_sum_kl_nats: float
    max_response_sum_kl_nats: float
    num_tokens: int
    num_responses: int
    clamped_tokens: int = 0

    def as_metrics(self) -> dict[str, float]:
        return {
            f"opd/{self.label}_k3_mean_token_nats": self.mean_token_kl_nats,
            f"opd/{self.label}_k3_response_sum_nats": self.mean_response_sum_kl_nats,
            f"opd/{self.label}_k3_max_response_sum_nats": self.max_response_sum_kl_nats,
            f"opd/{self.label}_k3_tokens": float(self.num_tokens),
            f"opd/{self.label}_k3_responses": float(self.num_responses),
            f"opd/{self.label}_k3_clamped_tokens": float(self.clamped_tokens),
        }


def k3_per_token(
    policy_logprobs: Sequence[float], ref_logprobs: Sequence[float]
) -> tuple[list[float], int]:
    """The nonnegative k3 estimator, per token, for samples drawn from the policy.

    ``logr = log p_ref - log p_policy``; ``k3 = exp(logr) - 1 - logr``. k3 is
    >= 0 for every logr and unbiased for ``KL(policy || ref)`` under sampling
    from the policy — unlike the signed k1 difference, which is negative about
    as often as it is positive on any individual token.

    Returns ``(per_token_k3, clamped_token_count)``.
    """
    if len(policy_logprobs) != len(ref_logprobs):
        raise AlignmentRefusal(
            REASON_BATCH_SHAPE,
            f"k3 needs matched arrays: policy has {len(policy_logprobs)} entries, "
            f"reference has {len(ref_logprobs)}",
        )
    out: list[float] = []
    clamped = 0
    for policy, ref in zip(policy_logprobs, ref_logprobs):
        logr = float(ref) - float(policy)
        if logr > K3_LOGR_CLAMP:
            logr = K3_LOGR_CLAMP
            clamped += 1
        out.append(math.exp(logr) - 1.0 - logr)
    return out, clamped


def kl_k3(
    label: str,
    policy_logprobs: Sequence[Sequence[float]],
    ref_logprobs: Sequence[Sequence[float]],
) -> KLTelemetry:
    """Mean token KL and response-summed KL in nats, over a batch of responses."""
    if len(policy_logprobs) != len(ref_logprobs):
        raise AlignmentRefusal(
            REASON_BATCH_SHAPE,
            f"k3 needs matched batches: {len(policy_logprobs)} policy responses vs "
            f"{len(ref_logprobs)} reference responses",
        )
    sums: list[float] = []
    total_tokens = 0
    total_kl = 0.0
    clamped = 0
    for policy, ref in zip(policy_logprobs, ref_logprobs):
        per_token, n_clamped = k3_per_token(policy, ref)
        clamped += n_clamped
        sums.append(sum(per_token))
        total_kl += sum(per_token)
        total_tokens += len(per_token)
    return KLTelemetry(
        label=label,
        mean_token_kl_nats=(total_kl / total_tokens) if total_tokens else 0.0,
        mean_response_sum_kl_nats=(sum(sums) / len(sums)) if sums else 0.0,
        max_response_sum_kl_nats=max(sums) if sums else 0.0,
        num_tokens=total_tokens,
        num_responses=len(sums),
        clamped_tokens=clamped,
    )


@dataclass(frozen=True)
class SignedTeacherSignal:
    """The signed teacher-minus-student log-probability signal, per token.

    This is the training objective's signal (the distillation advantage is
    ``kl_coefficient`` times this), and it is deliberately NOT a KL: it takes
    both signs. The cookbook logs the same quantity negated, as ``teacher_kl``
    / ``kl_policy_base``; those names are left untouched in the vendored tree
    and are not reused here.
    """

    per_token: tuple[tuple[float, ...], ...]
    mean_token_nats: float
    mean_response_sum_nats: float
    num_tokens: int
    num_responses: int

    def as_metrics(self) -> dict[str, float]:
        return {
            "opd/teacher_minus_student_k1_signed_mean_token_nats": self.mean_token_nats,
            "opd/teacher_minus_student_k1_signed_response_sum_nats": self.mean_response_sum_nats,
            # Same number negated: the sign convention the cookbook's signed
            # `teacher_kl` metric uses, kept so banked runs stay comparable.
            "opd/student_minus_teacher_k1_signed_mean_token_nats": -self.mean_token_nats,
            "opd/signed_signal_tokens": float(self.num_tokens),
            "opd/signed_signal_responses": float(self.num_responses),
        }


def teacher_minus_student(
    student_logprobs: Sequence[Sequence[float]],
    teacher_logprobs: Sequence[Sequence[float]],
) -> SignedTeacherSignal:
    """Per-token ``log p_teacher - log p_student`` over aligned completion spans."""
    if len(student_logprobs) != len(teacher_logprobs):
        raise AlignmentRefusal(
            REASON_BATCH_SHAPE,
            f"{len(student_logprobs)} student responses vs {len(teacher_logprobs)} teacher "
            "responses",
        )
    per_token: list[tuple[float, ...]] = []
    sums: list[float] = []
    total = 0.0
    n_tokens = 0
    for student, teacher in zip(student_logprobs, teacher_logprobs):
        if len(student) != len(teacher):
            raise AlignmentRefusal(
                REASON_BATCH_SHAPE,
                f"response has {len(student)} student log-probabilities and {len(teacher)} "
                "teacher log-probabilities; the completion spans are not aligned",
            )
        deltas = tuple(float(t) - float(s) for s, t in zip(student, teacher))
        per_token.append(deltas)
        sums.append(sum(deltas))
        total += sum(deltas)
        n_tokens += len(deltas)
    return SignedTeacherSignal(
        per_token=tuple(per_token),
        mean_token_nats=(total / n_tokens) if n_tokens else 0.0,
        mean_response_sum_nats=(sum(sums) / len(sums)) if sums else 0.0,
        num_tokens=n_tokens,
        num_responses=len(sums),
    )


def distillation_advantages(
    signal: SignedTeacherSignal, kl_coefficient: float
) -> list[list[float]]:
    """Per-token advantages: ``coef * (log p_teacher - log p_student)``.

    Identical in value to the stock recipe's advantage adjustment
    (``-coef * reverse_kl`` with ``reverse_kl = log p_student - log p_teacher``),
    so the training behavior of on-policy distillation is preserved exactly; the
    k3 metrics are additive monitoring, not a change of objective.
    """
    return [[kl_coefficient * v for v in response] for response in signal.per_token]


@dataclass(frozen=True)
class KLReport:
    """Everything the OPD loop logs about one batch's divergences."""

    signed: SignedTeacherSignal
    teacher_k3: KLTelemetry
    base_k3: KLTelemetry | None = None

    def as_metrics(self) -> dict[str, float]:
        out = dict(self.signed.as_metrics())
        out.update(self.teacher_k3.as_metrics())
        if self.base_k3 is not None:
            out.update(self.base_k3.as_metrics())
        return out


def kl_report(
    student_logprobs: Sequence[Sequence[float]],
    teacher_logprobs: Sequence[Sequence[float]],
    base_logprobs: Sequence[Sequence[float]] | None = None,
) -> KLReport:
    """Signed training signal + teacher k3 + (optional) base-reference k3.

    The policy is the student in both k3 estimates — the completions were
    sampled from it — so ``teacher_k3`` estimates ``KL(student || teacher)``
    (OPD convergence) and ``base_k3`` estimates ``KL(student || base)``, the
    quantity the DPO and RL arms are indexed by.
    """
    return KLReport(
        signed=teacher_minus_student(student_logprobs, teacher_logprobs),
        teacher_k3=kl_k3("teacher", student_logprobs, teacher_logprobs),
        base_k3=(
            None if base_logprobs is None else kl_k3("base_ref", student_logprobs, base_logprobs)
        ),
    )


# ---------------------------------------------------------------------------
# Single-response smoke gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmokeReport:
    """Proof that one real response aligned, with the numbers that prove it."""

    contract: RenderContract
    prompt: str
    completion_preview: str
    student_sequence_tokens: int
    teacher_sequence_tokens: int
    student_prefix_tokens: int
    teacher_prefix_tokens: int
    completion_tokens: int
    student_mask_sum: float
    teacher_mask_sum: float
    constitution_tokens: int
    max_sequence_tokens: int

    @property
    def ok(self) -> bool:
        return (
            self.completion_tokens > 0
            and self.student_mask_sum == self.completion_tokens
            and self.teacher_mask_sum == self.completion_tokens
            and self.constitution_tokens > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "prompt": self.prompt,
            "completion_preview": self.completion_preview,
            "student_sequence_tokens": self.student_sequence_tokens,
            "teacher_sequence_tokens": self.teacher_sequence_tokens,
            "student_prefix_tokens": self.student_prefix_tokens,
            "teacher_prefix_tokens": self.teacher_prefix_tokens,
            "completion_tokens": self.completion_tokens,
            "student_mask_sum": self.student_mask_sum,
            "teacher_mask_sum": self.teacher_mask_sum,
            "constitution_tokens": self.constitution_tokens,
            "max_sequence_tokens": self.max_sequence_tokens,
            "contract": self.contract.to_dict(),
        }


def single_response_smoke(
    renderer: AsymmetricRenderer,
    pair: PromptPair,
    completion: str,
    *,
    sampled_tokens: Sequence[int] | None = None,
    max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS,
) -> SmokeReport:
    """Prove aligned token counts and loss masks on ONE response.

    Required before any paid training request (readiness doc, OPD gap item).
    Raises :class:`AlignmentRefusal` if the response does not align; returns a
    report whose numbers are the evidence, not a boolean assertion of it.
    """
    aligned = align_pair(
        renderer,
        pair,
        completion,
        sampled_tokens=sampled_tokens,
        max_sequence_tokens=max_sequence_tokens,
    )
    student_mask = sum(aligned.target_mask("student"))
    teacher_mask = sum(aligned.target_mask("teacher"))
    if student_mask != teacher_mask or student_mask != aligned.num_completion_tokens:
        raise AlignmentRefusal(
            REASON_LOSS_MASK_MISMATCH,
            f"loss masks select {student_mask} student and {teacher_mask} teacher tokens for "
            f"a {aligned.num_completion_tokens}-token completion",
        )
    report = SmokeReport(
        contract=aligned.contract,
        prompt=pair.prompt,
        completion_preview=completion[:200],
        student_sequence_tokens=len(aligned.student_tokens),
        teacher_sequence_tokens=len(aligned.teacher_tokens),
        student_prefix_tokens=aligned.student_prefix_len,
        teacher_prefix_tokens=aligned.teacher_prefix_len,
        completion_tokens=aligned.num_completion_tokens,
        student_mask_sum=student_mask,
        teacher_mask_sum=teacher_mask,
        constitution_tokens=aligned.constitution_tokens,
        max_sequence_tokens=max_sequence_tokens,
    )
    if not report.ok:
        raise AlignmentRefusal(
            REASON_LOSS_MASK_MISMATCH,
            f"smoke report did not satisfy its own gate: {report.to_dict()}",
        )
    return report


def require_smoke_gate(report: SmokeReport | None, contract: RenderContract) -> SmokeReport:
    """Fail closed before a paid training request without a passing smoke."""
    if report is None:
        raise SmokeGateError(
            "OPD refuses to issue a paid training request without a single-response "
            "alignment smoke; run single_response_smoke() first"
        )
    if not report.ok:
        raise SmokeGateError(f"OPD smoke did not pass: {report.to_dict()}")
    if report.contract != contract:
        raise SmokeGateError(
            f"OPD smoke was gated against contract {report.contract.instrument_id} but the "
            f"run renders with {contract.instrument_id}"
        )
    return report


# ---------------------------------------------------------------------------
# Dry-run plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OPDPlan:
    """Request/token plan for one OPD run — the free, pre-spend view."""

    config: OPDConfig
    num_prompts: int
    steps: int
    samples: int
    student_sample_tokens: int
    teacher_logprob_tokens: int
    base_logprob_tokens: int
    train_tokens: int
    checkpoints: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": vars(self.config),
            "num_prompts": self.num_prompts,
            "steps": self.steps,
            "samples": self.samples,
            "student_sample_tokens": self.student_sample_tokens,
            "teacher_logprob_tokens": self.teacher_logprob_tokens,
            "base_logprob_tokens": self.base_logprob_tokens,
            "train_tokens": self.train_tokens,
            "checkpoints": self.checkpoints,
            "config_hash": manifest.config_hash(self.config),
        }


def plan(
    config: OPDConfig = OPD_PILOT,
    *,
    num_prompts: int,
    prompt_tokens: int = 512,
    constitution_tokens: int = 1024,
) -> OPDPlan:
    """Pessimistic (max-envelope) request/token plan; no network, no spend."""
    steps = min(config.max_steps, max(1, num_prompts // config.prompts_per_batch))
    samples = steps * config.samples_per_step
    response = config.max_response_tokens
    student_seq = prompt_tokens + response
    teacher_seq = constitution_tokens + prompt_tokens + response
    return OPDPlan(
        config=config,
        num_prompts=num_prompts,
        steps=steps,
        samples=samples,
        student_sample_tokens=samples * response,
        teacher_logprob_tokens=samples * teacher_seq,
        base_logprob_tokens=samples * student_seq,
        train_tokens=samples * student_seq,
        checkpoints=max(1, steps // max(1, config.save_every)),
    )


# ---------------------------------------------------------------------------
# Run orchestration (dry-run by default)
# ---------------------------------------------------------------------------


def run(
    constitution: Constitution,
    prompts: Sequence[str],
    out_dir: Path,
    runtime: TinkerRuntime,
    config: OPDConfig = OPD_PILOT,
    *,
    execute: bool = False,
) -> dict[str, Any]:
    """Run OPD for one persona. Dry-run by default; ``execute`` spends money.

    The dry-run path writes ``opd_plan.json`` (request/token plan) and touches
    no client. The paid path additionally requires a passing single-response
    alignment smoke before its first training request.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opd_plan = plan(config, num_prompts=len(prompts))

    if not execute or runtime.config.dry_run:
        payload = {
            "status": "dry-run",
            "persona": constitution.persona,
            "plan": opd_plan.to_dict(),
            "note": "no Tinker calls were made; pass execute=True with a live runtime to spend",
        }
        manifest.atomic_write_json(out_dir / "opd_plan.json", payload)
        logger.info(
            "OPD dry run: %d steps x %d samples (%d student sample tokens planned)",
            opd_plan.steps,
            config.samples_per_step,
            opd_plan.student_sample_tokens,
        )
        return payload

    return _run_opd_real(constitution, prompts, out_dir, runtime, config, opd_plan)


async def sample_group(
    client: Any,
    model_input: Any,
    *,
    num_samples: int,
    max_tokens: int,
    temperature: float,
    stop: Any,
) -> list[tuple[list[int], list[float]]]:  # pragma: no cover - paid path
    """Sample ``num_samples`` completions, returning ids AND their log-probs.

    The sampler's per-token log-probabilities are the importance weights of the
    training step, so they are carried alongside the ids rather than recomputed.
    """
    import tinker

    result = await client.sample_async(
        prompt=model_input,
        num_samples=num_samples,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, stop=stop
        ),
    )
    return [
        ([int(t) for t in seq.tokens], [float(lp) for lp in seq.logprobs])
        for seq in result.sequences
    ]


def _completion_text(handle: RendererHandle, tokens: Sequence[int]) -> str:  # pragma: no cover
    """Visible text of a sampled completion, normalized as every other stage does."""
    message, _termination = handle.renderer.parse_response(list(tokens))
    content = message.get("content", "")
    return generation._clean_completion(generation._visible_text(content))


def _run_opd_real(  # pragma: no cover - paid path, exercised by the smoke gate
    constitution: Constitution,
    prompts: Sequence[str],
    out_dir: Path,
    runtime: TinkerRuntime,
    config: OPDConfig,
    opd_plan: OPDPlan,
) -> dict[str, Any]:
    """The paid OPD loop: sample on-policy, score the teacher asymmetrically, step.

    Mirrors the stock recipe's objective (advantage = coefficient times the
    signed teacher-minus-student log-probability, importance-sampling loss) but
    reads the teacher's log-probabilities from ITS OWN sequence via the aligned
    span, which is the gap this module exists to close.
    """
    import asyncio

    import tinker
    from tinker_cookbook import checkpoint_utils

    service_client = runtime.require_service_client()
    renderer = AsymmetricRenderer.from_runtime(
        runtime, config.student_model, config.teacher_model
    )
    contract = renderer.contract()
    pairs = constitution_prompt_pairs(constitution, prompts, student_model=config.student_model)
    stop = renderer.student.renderer.get_stop_sequences()

    teacher_client = service_client.create_sampling_client(
        base_model=config.teacher_model, model_path=config.teacher_checkpoint
    )
    base_client = service_client.create_sampling_client(
        base_model=config.student_model, model_path=config.base_reference_checkpoint
    )
    training_client = service_client.create_lora_training_client(
        base_model=config.student_model, rank=config.lora_rank
    )
    sampling_client = training_client.save_weights_and_get_sampling_client()

    # --- gate: one real response must align before any training request -----
    smoke_pair = pairs[0]
    smoke_tokens, _smoke_logprobs = asyncio.run(
        sample_group(
            sampling_client,
            renderer.student.renderer.build_generation_prompt(list(smoke_pair.student_messages)),
            num_samples=1,
            max_tokens=config.max_response_tokens,
            temperature=config.temperature,
            stop=stop,
        )
    )[0]
    smoke = single_response_smoke(
        renderer,
        smoke_pair,
        _completion_text(renderer.student, smoke_tokens),
        sampled_tokens=smoke_tokens,
        max_sequence_tokens=config.max_sequence_tokens,
    )
    require_smoke_gate(smoke, contract)
    manifest.atomic_write_json(out_dir / "opd_smoke.json", smoke.to_dict())
    logger.info("OPD alignment smoke passed: %s", smoke.to_dict())

    checkpoint_mgr = checkpoint_utils.CheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=str(out_dir),
        save_every=config.save_every,
        ttl_seconds=None,
    )

    for step in range(config.max_steps):
        batch_pairs = [
            pairs[(step * config.prompts_per_batch + i) % len(pairs)]
            for i in range(config.prompts_per_batch)
        ]

        async def _sample_batch(batch=batch_pairs, client=sampling_client):
            return await asyncio.gather(
                *[
                    sample_group(
                        client,
                        renderer.student.renderer.build_generation_prompt(
                            list(pair.student_messages)
                        ),
                        num_samples=config.samples_per_prompt,
                        max_tokens=config.max_response_tokens,
                        temperature=config.temperature,
                        stop=stop,
                    )
                    for pair in batch
                ]
            )

        groups = asyncio.run(_sample_batch())
        flat_pairs: list[PromptPair] = []
        flat_tokens: list[list[int]] = []
        flat_logprobs: list[list[float]] = []
        for pair, group in zip(batch_pairs, groups):
            for tokens, logprobs in group:
                flat_pairs.append(pair)
                flat_tokens.append(tokens)
                flat_logprobs.append(logprobs)

        completions = [_completion_text(renderer.student, t) for t in flat_tokens]
        batch = align_batch(
            renderer,
            flat_pairs,
            completions,
            sampled_tokens=flat_tokens,
            max_sequence_tokens=config.max_sequence_tokens,
            expected_contract=contract,
        )

        async def _score(aligned=batch):
            teacher = asyncio.gather(
                *[
                    teacher_client.compute_logprobs_async(
                        tinker.ModelInput.from_ints(list(p.teacher_tokens))
                    )
                    for p in aligned.pairs
                ]
            )
            base = asyncio.gather(
                *[
                    base_client.compute_logprobs_async(
                        tinker.ModelInput.from_ints(list(p.student_tokens))
                    )
                    for p in aligned.pairs
                ]
            )
            return await asyncio.gather(teacher, base)

        teacher_full, base_full = asyncio.run(_score())
        teacher_lp = [
            p.completion_logprobs(lp, side="teacher")
            for p, lp in zip(batch.pairs, teacher_full)
        ]
        base_lp = [
            p.completion_logprobs(lp, side="student") for p, lp in zip(batch.pairs, base_full)
        ]
        student_lp = list(flat_logprobs)

        report = kl_report(student_lp, teacher_lp, base_lp)
        advantages = distillation_advantages(report.signed, config.kl_coefficient)

        # The loss mask is client-side bookkeeping: the stock loop strips it
        # before the request (`rl.train._remove_mask`) and relies on advantages
        # being zero off the action span. Both are asserted here rather than
        # assumed, because a nonzero advantage on a prompt token would train the
        # student on the teacher's opinion of ITS OWN prompt.
        data = []
        for p, lp, adv in zip(batch.pairs, student_lp, advantages):
            mask = p.target_mask("student")
            scattered_lp = p.scatter_over_targets(lp)
            scattered_adv = p.scatter_over_targets(adv)
            off_span = [i for i, m in enumerate(mask) if m == 0.0]
            if any(scattered_adv[i] != 0.0 or scattered_lp[i] != 0.0 for i in off_span):
                raise AlignmentRefusal(
                    REASON_LOSS_MASK_MISMATCH,
                    "advantages or sampled log-probabilities are nonzero outside the "
                    "completion span",
                )
            data.append(
                tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(list(p.student_tokens[:-1])),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData.from_numpy(
                            _np_array(list(p.student_tokens[1:]), dtype="int64")
                        ),
                        "logprobs": tinker.TensorData.from_numpy(
                            _np_array(scattered_lp, dtype="float32")
                        ),
                        "advantages": tinker.TensorData.from_numpy(
                            _np_array(scattered_adv, dtype="float32")
                        ),
                    },
                )
            )

        training_client.forward_backward(data, loss_fn="importance_sampling").result()
        training_client.optim_step(
            tinker.AdamParams(
                learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
            )
        ).result()

        checkpoint_mgr.maybe_save(step=step + 1, loop_state={"batch": step + 1})
        sampling_client = training_client.save_weights_and_get_sampling_client()

        row = {
            "step": step,
            "lr": config.learning_rate,
            "num_samples": len(batch),
            "completion_tokens": batch.num_completion_tokens,
            "instrument_id": contract.instrument_id,
            "render_contract": contract.contract_version,
            # The pilot's save/evaluate cadence. Saving happens here; the
            # OUT-OF-LOOP character/coherence evaluation is not wired in this
            # module (it belongs to octt.evaluation's instruments), so the
            # cadence is stamped for whoever indexes those runs by step.
            "eval_checkpoint": step % config.eval_every == 0,
            **report.as_metrics(),
        }
        with open(out_dir / "metrics.jsonl", "a") as metrics_file:
            metrics_file.write(json.dumps(row) + "\n")
        logger.info(
            "OPD step %d: teacher k3 %.4f nats/token, base k3 %.4f nats/token",
            step,
            report.teacher_k3.mean_token_kl_nats,
            report.base_k3.mean_token_kl_nats if report.base_k3 else float("nan"),
        )

    paths = checkpoint_mgr.save_final(loop_state={"batch": config.max_steps})
    record = checkpoint_utils.get_last_checkpoint(str(out_dir), required_key="sampler_path")
    checkpoint = manifest.StageCheckpoint(
        sampler_path=paths.get("sampler_path") or (record.sampler_path if record else None),
        state_path=paths.get("state_path") or (record.state_path if record else None),
        step=config.max_steps,
        config_hash=manifest.config_hash(config),
        extra={"stage": "opd", "contract": contract.to_dict()},
    )
    payload = {
        "status": "executed",
        "persona": constitution.persona,
        "plan": opd_plan.to_dict(),
        "smoke": smoke.to_dict(),
        "checkpoint": checkpoint.to_dict(),
    }
    manifest.atomic_write_json(out_dir / "opd_run.json", payload)
    return payload


def _np_array(values: Sequence[Any], *, dtype: str) -> Any:  # pragma: no cover - paid path
    import numpy as np

    return np.array(values, dtype=dtype)


__all__ = [
    "DEFAULT_MAX_SEQUENCE_TOKENS",
    "OPD_PILOT",
    "RENDER_CONTRACT_VERSION",
    "AlignedBatch",
    "AlignedPair",
    "AlignmentRefusal",
    "AsymmetricRenderer",
    "KLReport",
    "KLTelemetry",
    "OPDConfig",
    "OPDPlan",
    "PromptPair",
    "RenderContract",
    "RendererHandle",
    "SignedTeacherSignal",
    "SmokeGateError",
    "SmokeReport",
    "align_batch",
    "align_pair",
    "constitution_prompt_pairs",
    "distillation_advantages",
    "k3_per_token",
    "kl_k3",
    "kl_report",
    "plan",
    "prompt_pair",
    "require_smoke_gate",
    "run",
    "single_response_smoke",
    "teacher_minus_student",
]
