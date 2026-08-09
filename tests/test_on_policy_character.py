"""Tests for the OPD asymmetric-context alignment layer.

Every alignment test proves one of exactly two outcomes: the layer aligns the
completion span EXACTLY, or it refuses. There is no assertion anywhere in this
file that tolerates an approximate or partial alignment, because the failure
mode being defended against is silent: a misaligned teacher slice trains the
student against log-probabilities for the wrong tokens and the run still looks
healthy.

The tokenizer here is a deterministic fake, never a downloaded one. It is not a
stub: it does greedy longest-match over an explicit vocabulary, so it can merge
a token across the prefix/completion boundary the way real BPE can, which is
what makes the refusal paths reachable offline.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from octt import on_policy_character as opd
from octt.constitution import Constitution
from octt.tinker_client import TinkerClientConfig, TinkerRuntime

# ---------------------------------------------------------------------------
# Deterministic fake tokenizer / renderer
# ---------------------------------------------------------------------------

CHAR_BASE = 1000
SPECIALS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")


class FakeTokenizer:
    """Greedy longest-match tokenizer over an explicit vocabulary.

    Ids below ``CHAR_BASE`` are vocabulary pieces (specials + merges); every
    other character maps to ``CHAR_BASE + ord(c)``. Fully deterministic and
    dependency-free, and — unlike a whitespace splitter — capable of producing
    a token that spans a rendered boundary.
    """

    def __init__(self, *, merges: tuple[str, ...] = (), name: str = "fake-v1") -> None:
        self.name_or_path = name
        self.pieces = list(SPECIALS) + list(merges)
        assert len(self.pieces) < CHAR_BASE
        self._order = sorted(range(len(self.pieces)), key=lambda i: (-len(self.pieces[i]), i))

    @property
    def vocab_size(self) -> int:
        return CHAR_BASE + len(self.pieces)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            for idx in self._order:
                piece = self.pieces[idx]
                if text.startswith(piece, i):
                    out.append(idx)
                    i += len(piece)
                    break
            else:
                out.append(CHAR_BASE + ord(text[i]))
                i += 1
        return out

    def decode(self, tokens: list[int]) -> str:
        return "".join(
            self.pieces[t] if t < CHAR_BASE else chr(t - CHAR_BASE) for t in tokens
        )


class FakeChatRenderer:
    """A ChatML-shaped renderer that tokenizes the WHOLE rendered string.

    Text-first rather than chunk-first (the stricter of the two models a real
    renderer can follow): tokens may merge across the prefix/completion
    boundary, so a renderer built this way genuinely can misalign, and the
    layer has to catch it rather than be structurally unable to fail.

    ``quirk`` reproduces a specific real-renderer misbehavior:

    - ``thinking_prompt``  generation prompt opens ``<think>`` that the
      supervised render does not reproduce (thinking-renderer mismatch);
    - ``drop_system``      system messages are dropped from the render;
    - ``system_shifts_completion``  the assistant message is rendered
      differently when a system block is present;
    - ``drop_completion``  the assistant message is not rendered at all;
    - ``weight_prompt``    loss weights cover the prompt as well.
    """

    def __init__(self, tokenizer: FakeTokenizer, *, quirk: str | None = None) -> None:
        self.tokenizer = tokenizer
        self.quirk = quirk

    def _history_text(self, messages) -> str:
        parts = []
        for message in messages:
            if self.quirk == "drop_system" and message["role"] == "system":
                continue
            parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def build_generation_prompt(self, messages) -> list[int]:
        text = self._history_text(messages)
        if self.quirk == "thinking_prompt":
            text += "<think>\n"
        return self.tokenizer.encode(text)

    def build_supervised_example(self, messages, train_on_what=None):
        history, assistant = list(messages[:-1]), messages[-1]
        base = self._history_text(history)
        if self.quirk == "thinking_prompt":
            base += "<think>\n"
        content = str(assistant["content"])
        if self.quirk == "system_shifts_completion" and any(
            m["role"] == "system" for m in history
        ):
            content = " " + content
        if self.quirk == "drop_completion":
            full_text = base
        else:
            full_text = f"{base}{content}<|im_end|>\n"
        if self.quirk == "thinking_prompt":
            # The supervised render forgets the generation-time <think> opener.
            full_text = full_text.replace("<think>\n", "", 1)
        full = self.tokenizer.encode(full_text)
        prefix_len = min(len(self.tokenizer.encode(base)), len(full))
        weights = [0.0] * prefix_len + [1.0] * (len(full) - prefix_len)
        if self.quirk == "weight_prompt":
            weights = [1.0] * len(full)
        return full, weights


class OtherFakeChatRenderer(FakeChatRenderer):
    """A different renderer CLASS on the same template — a contract change."""


def make_renderer(
    *,
    student_quirk: str | None = None,
    teacher_quirk: str | None = None,
    merges: tuple[str, ...] = (),
) -> opd.AsymmetricRenderer:
    tokenizer = FakeTokenizer(merges=merges)
    return opd.AsymmetricRenderer(
        student=opd.RendererHandle(
            model_id="Qwen/Qwen3.5-4B",
            renderer_name="fake_chatml",
            renderer=FakeChatRenderer(tokenizer, quirk=student_quirk),
            tokenizer=tokenizer,
        ),
        teacher=opd.RendererHandle(
            model_id="Qwen/Qwen3.5-4B",
            renderer_name="fake_chatml",
            renderer=FakeChatRenderer(tokenizer, quirk=teacher_quirk),
            tokenizer=tokenizer,
        ),
    )


CONSTITUTION_TEXT = "I am playful and direct. I never pretend to certainty I lack."


def make_pair(prompt: str = "What is a good breakfast?") -> opd.PromptPair:
    return opd.prompt_pair(prompt, CONSTITUTION_TEXT)


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_module_imports_without_the_training_stack():
    """The package must import with no tinker/torch/transformers present."""
    code = (
        "import sys; import octt.on_policy_character as m; "
        "assert m.OPD_PILOT.lora_rank == 32; "
        "leaked = [n for n in ('tinker', 'torch', 'transformers', 'peft') "
        "if n in sys.modules]; "
        "assert not leaked, leaked; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# ---------------------------------------------------------------------------
# Pilot configuration
# ---------------------------------------------------------------------------


def test_pilot_config_encodes_the_documented_numbers():
    cfg = opd.OPD_PILOT
    assert cfg.student_model == "Qwen/Qwen3.5-4B"
    assert cfg.teacher_model == "Qwen/Qwen3.5-4B"
    assert cfg.lora_rank == 32
    assert cfg.learning_rate == 1e-4
    assert (cfg.prompts_per_batch, cfg.samples_per_prompt) == (8, 4)
    assert cfg.samples_per_step == 32
    assert cfg.temperature == 1.0
    assert cfg.max_response_tokens == 512
    assert cfg.kl_coefficient == 1.0
    assert cfg.max_steps == 20
    assert cfg.save_every == 5
    assert cfg.eval_every == 5


def test_config_rejects_degenerate_settings():
    with pytest.raises(ValueError):
        opd.OPDConfig(samples_per_prompt=0)
    with pytest.raises(ValueError):
        opd.OPDConfig(temperature=0.0)


# ---------------------------------------------------------------------------
# Prompt mapping
# ---------------------------------------------------------------------------


def test_prompt_mapping_gives_the_constitution_to_the_teacher_only():
    pair = make_pair("hello")
    assert [m["role"] for m in pair.student_messages] == ["user"]
    assert [m["role"] for m in pair.teacher_messages] == ["system", "user"]
    assert pair.teacher_messages[0]["content"] == CONSTITUTION_TEXT
    assert all(CONSTITUTION_TEXT not in str(m["content"]) for m in pair.student_messages)
    # The user turn is byte-identical in both views.
    assert pair.student_messages[-1] == pair.teacher_messages[-1]


def test_constitution_prompt_pairs_use_the_paper_character_prompt():
    constitution = Constitution(persona="pirate", assertions=("I am bold.", "I love the sea."))
    pairs = opd.constitution_prompt_pairs(
        constitution, ["hi", "hello"], student_model="Qwen/Qwen3.5-4B"
    )
    assert len(pairs) == 2
    assert "I am bold." in pairs[0].system
    assert "Qwen" in pairs[0].system  # assistant_name for the Qwen family


# ---------------------------------------------------------------------------
# Happy-path alignment: exact, on both sides
# ---------------------------------------------------------------------------


ALIGNING_COMPLETIONS = [
    pytest.param("Oatmeal, and plenty of it.", id="ascii"),
    pytest.param("Café crème, 早餐, and a 🥐 — naturally.", id="unicode-mixed"),
    pytest.param("Ω≈ç√∫˜µ≤≥÷ и наконец, ελληνικά", id="unicode-symbols"),
    pytest.param("égg (combining acute) and eé precomposed", id="unicode-combining"),
    pytest.param("line one\n\n\tline two   with  runs\n", id="whitespace-runs"),
    pytest.param("   leading and trailing   ", id="whitespace-edges"),
    pytest.param('<tool_call>{"name": "eggs", "args": {"n": 2}}</tool_call>', id="tool-call"),
    pytest.param("a < b and c > d, not <|a control token", id="angle-brackets"),
    pytest.param("x", id="single-char"),
]


@pytest.mark.parametrize("completion", ALIGNING_COMPLETIONS)
def test_alignment_is_exact_for_wellformed_completions(completion):
    renderer = make_renderer()
    pair = make_pair()

    aligned = opd.align_pair(renderer, pair, completion)

    # The located spans are token-for-token identical...
    student_span = aligned.student_tokens[aligned.student_prefix_len :]
    teacher_span = aligned.teacher_tokens[aligned.teacher_prefix_len :]
    assert student_span == teacher_span == aligned.completion_tokens
    assert aligned.num_completion_tokens > 0
    # ...at DIFFERENT offsets, which is the whole point of the layer.
    assert aligned.teacher_prefix_len > aligned.student_prefix_len
    assert aligned.constitution_tokens == (
        aligned.teacher_prefix_len - aligned.student_prefix_len
    )
    # ...and the prefixes really are prefixes of the full renders.
    assert aligned.student_tokens[: aligned.student_prefix_len] == renderer.student.prefix_tokens(
        pair.student_messages
    )
    assert aligned.teacher_tokens[: aligned.teacher_prefix_len] == renderer.teacher.prefix_tokens(
        pair.teacher_messages
    )


def test_loss_masks_select_exactly_the_completion_on_both_sides():
    renderer = make_renderer()
    aligned = opd.align_pair(renderer, make_pair(), "Porridge. 🥣")

    for side in ("student", "teacher"):
        mask = aligned.target_mask(side)
        tokens = aligned.tokens(side)
        prefix = aligned.prefix_len(side)
        assert len(mask) == len(tokens) - 1
        assert sum(mask) == aligned.num_completion_tokens
        # Entry j scores the prediction of tokens[j + 1].
        assert all(mask[j] == 0.0 for j in range(prefix - 1))
        assert all(mask[j] == 1.0 for j in range(prefix - 1, len(mask)))


def test_completion_logprobs_slice_the_right_indices_on_each_side():
    renderer = make_renderer()
    aligned = opd.align_pair(renderer, make_pair(), "Two eggs.")

    # Log-probability arrays whose value IS the index, so a slice can be checked
    # against the positions it should have taken.
    student_lp = [float(i) for i in range(len(aligned.student_tokens))]
    teacher_lp = [float(i) for i in range(len(aligned.teacher_tokens))]

    student_slice = aligned.completion_logprobs(student_lp, side="student")
    teacher_slice = aligned.completion_logprobs(teacher_lp, side="teacher")

    n = aligned.num_completion_tokens
    assert len(student_slice) == len(teacher_slice) == n
    assert student_slice == [float(aligned.student_prefix_len + i) for i in range(n)]
    assert teacher_slice == [float(aligned.teacher_prefix_len + i) for i in range(n)]
    # The teacher slice starts later by exactly the constitution's token count.
    assert teacher_slice[0] - student_slice[0] == float(aligned.constitution_tokens)


def test_completion_logprobs_refuse_a_wrong_length_array():
    renderer = make_renderer()
    aligned = opd.align_pair(renderer, make_pair(), "Toast.")
    # The classic bug: feeding the teacher's array to the student's slice.
    teacher_lp = [0.0] * len(aligned.teacher_tokens)
    with pytest.raises(opd.AlignmentRefusal) as exc:
        aligned.completion_logprobs(teacher_lp, side="student")
    assert exc.value.reason == opd.REASON_BATCH_SHAPE


def test_scatter_over_targets_places_values_on_the_completion_span():
    renderer = make_renderer()
    aligned = opd.align_pair(renderer, make_pair(), "Beans.")
    values = [float(i + 1) for i in range(aligned.num_completion_tokens)]

    scattered = aligned.scatter_over_targets(values, side="student")

    mask = aligned.target_mask("student")
    assert len(scattered) == len(mask)
    assert [v for v, m in zip(scattered, mask) if m] == values
    assert all(v == 0.0 for v, m in zip(scattered, mask) if not m)


def test_sampled_tokens_matching_the_render_are_accepted():
    renderer = make_renderer()
    pair = make_pair()
    completion = "Congee, with ginger."
    reference = opd.align_pair(renderer, pair, completion)

    aligned = opd.align_pair(
        renderer, pair, completion, sampled_tokens=reference.completion_tokens
    )
    assert aligned.completion_tokens == reference.completion_tokens


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "completion", ["", " ", "\n", "\t \n  ", " "], ids=["empty", "space", "nl", "mixed", "nbsp"]
)
def test_empty_or_whitespace_completions_are_refused(completion):
    renderer = make_renderer()
    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.align_pair(renderer, make_pair(), completion)
    assert exc.value.reason == opd.REASON_EMPTY_COMPLETION


def test_renderer_that_emits_no_completion_tokens_is_refused():
    """An assistant message the renderer drops must not read as a zero-token span."""
    renderer = make_renderer(student_quirk="drop_completion", teacher_quirk="drop_completion")
    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.align_pair(renderer, make_pair(), "this text never gets rendered")
    assert exc.value.reason == opd.REASON_PREFIX_NOT_A_PREFIX
    assert "no completion tokens were located" in exc.value.detail


def test_completion_containing_a_control_token_is_refused():
    renderer = make_renderer()
    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.align_pair(renderer, make_pair(), "Eggs<|im_end|>and then some")
    assert exc.value.reason == opd.REASON_CONTROL_TOKEN_IN_COMPLETION


def test_token_merging_across_the_prefix_boundary_is_refused():
    """A vocabulary piece spanning the header's newline and the completion's
    first character makes the generation prompt stop being a prefix."""
    renderer = make_renderer(merges=("\n🙂",))
    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.align_pair(renderer, make_pair(), "🙂 good morning")
    assert exc.value.reason == opd.REASON_PREFIX_NOT_A_PREFIX


def test_whitespace_merge_across_the_boundary_is_refused():
    renderer = make_renderer(merges=("\n   ",))
    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.align_pair(renderer, make_pair(), "   indented answer")
    assert exc.value.reason == opd.REASON_PREFIX_NOT_A_PREFIX


def test_teacher_context_changing_the_completion_tokenization_is_refused():
    """The failure this layer exists for: the same text, different tokens."""
    renderer = make_renderer(teacher_quirk="system_shifts_completion")
    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.align_pair(renderer, make_pair(), "Shakshuka.")
    assert exc.value.reason == opd.REASON_COMPLETION_MISMATCH


def test_generation_prompt_that_is_not_a_prefix_is_refused():
    """A thinking renderer whose generation prompt opens a block the supervised
    render omits: the completion span cannot be located by subtraction."""
    renderer = make_renderer(student_quirk="thinking_prompt", teacher_quirk="thinking_prompt")
    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.align_pair(renderer, make_pair(), "Muesli.")
    assert exc.value.reason == opd.REASON_PREFIX_NOT_A_PREFIX


def test_renderer_that_drops_the_system_block_is_refused():
    """Dropping the constitution makes OPD a no-op that still 'trains'."""
    renderer = make_renderer(student_quirk="drop_system", teacher_quirk="drop_system")
    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.align_pair(renderer, make_pair(), "Grits.")
    assert exc.value.reason == opd.REASON_CONSTITUTION_ABSENT


def test_loss_weights_covering_the_prompt_are_refused():
    renderer = make_renderer(student_quirk="weight_prompt", teacher_quirk="weight_prompt")
    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.align_pair(renderer, make_pair(), "Kedgeree.")
    assert exc.value.reason == opd.REASON_LOSS_MASK_MISMATCH


def test_sampled_tokens_that_disagree_with_the_render_are_refused():
    renderer = make_renderer()
    pair = make_pair()
    completion = "Bagels."
    reference = opd.align_pair(renderer, pair, completion)
    tampered = list(reference.completion_tokens)[:-1]

    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.align_pair(renderer, pair, completion, sampled_tokens=tampered)
    assert exc.value.reason == opd.REASON_SAMPLED_TOKEN_MISMATCH


def test_over_length_teacher_sequence_is_refused_not_truncated():
    """The teacher carries the constitution, so it overflows first. Truncating
    it would drop the tail of the completion it is supposed to be scoring."""
    renderer = make_renderer()
    long_system = "I am relentlessly verbose. " * 40
    pair = opd.prompt_pair("Breakfast?", long_system)
    completion = "Shakshuka, slowly."

    fits = opd.align_pair(renderer, pair, completion, max_sequence_tokens=100_000)
    limit = len(fits.student_tokens) + 10
    assert limit < len(fits.teacher_tokens)  # the student alone would have fit

    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.align_pair(renderer, pair, completion, max_sequence_tokens=limit)
    assert exc.value.reason == opd.REASON_TRUNCATED
    assert "teacher" in exc.value.detail


# ---------------------------------------------------------------------------
# Renderer-version mismatch
# ---------------------------------------------------------------------------


def test_two_different_renderer_names_cannot_form_a_pair():
    tokenizer = FakeTokenizer()
    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.AsymmetricRenderer(
            student=opd.RendererHandle(
                "Qwen/Qwen3.5-4B", "qwen3_5_disable_thinking", FakeChatRenderer(tokenizer),
                tokenizer,
            ),
            teacher=opd.RendererHandle(
                "Qwen/Qwen3.5-4B", "qwen3_5", FakeChatRenderer(tokenizer), tokenizer
            ),
        )
    assert exc.value.reason == opd.REASON_RENDERER_MISMATCH


def test_two_different_renderer_classes_cannot_form_a_pair():
    tokenizer = FakeTokenizer()
    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.AsymmetricRenderer(
            student=opd.RendererHandle(
                "Qwen/Qwen3.5-4B", "fake_chatml", FakeChatRenderer(tokenizer), tokenizer
            ),
            teacher=opd.RendererHandle(
                "Qwen/Qwen3.5-4B", "fake_chatml", OtherFakeChatRenderer(tokenizer), tokenizer
            ),
        )
    assert exc.value.reason == opd.REASON_RENDERER_MISMATCH


def test_two_different_tokenizers_cannot_form_a_pair():
    student_tok = FakeTokenizer()
    teacher_tok = FakeTokenizer(merges=("breakfast",))
    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.AsymmetricRenderer(
            student=opd.RendererHandle(
                "Qwen/Qwen3.5-4B", "fake_chatml", FakeChatRenderer(student_tok), student_tok
            ),
            teacher=opd.RendererHandle(
                "Qwen/Qwen3.5-4B", "fake_chatml", FakeChatRenderer(teacher_tok), teacher_tok
            ),
        )
    assert exc.value.reason == opd.REASON_RENDERER_MISMATCH


def test_tokenizer_differing_only_inside_the_probe_is_caught():
    """The fingerprint's probe does real work, not just its identity fields:
    these two vocabularies are the same size and name, and differ only in a
    merge the probe exercises."""
    student_tok = FakeTokenizer(merges=("wörld",))
    teacher_tok = FakeTokenizer(merges=("zzzzz",))
    assert student_tok.vocab_size == teacher_tok.vocab_size
    assert student_tok.name_or_path == teacher_tok.name_or_path

    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.AsymmetricRenderer(
            student=opd.RendererHandle(
                "Qwen/Qwen3.5-4B", "fake_chatml", FakeChatRenderer(student_tok), student_tok
            ),
            teacher=opd.RendererHandle(
                "Qwen/Qwen3.5-4B", "fake_chatml", FakeChatRenderer(teacher_tok), teacher_tok
            ),
        )
    assert exc.value.reason == opd.REASON_RENDERER_MISMATCH


def test_contract_drift_after_the_gate_is_refused():
    """The renderer changed between the smoke gate and the training batch."""
    gated = make_renderer().contract()
    drifted = make_renderer(merges=("breakfast",))

    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.align_pair(drifted, make_pair(), "Eggs.", expected_contract=gated)
    assert exc.value.reason == opd.REASON_RENDERER_MISMATCH
    assert drifted.contract() != gated


def test_contract_is_stable_across_equivalent_renderers():
    assert make_renderer().contract() == make_renderer().contract()
    assert make_renderer().contract().contract_version == opd.RENDER_CONTRACT_VERSION
    assert make_renderer().contract().instrument_id


# ---------------------------------------------------------------------------
# Batch behavior: one bad sample refuses the whole batch
# ---------------------------------------------------------------------------


def test_batch_aligns_when_every_sample_aligns():
    renderer = make_renderer()
    pairs = [make_pair("q1"), make_pair("q2"), make_pair("q3")]
    completions = ["one 🙂", "two\n\ttabbed", "three"]

    batch = opd.align_batch(renderer, pairs, completions)

    assert len(batch) == 3
    assert batch.num_completion_tokens == sum(p.num_completion_tokens for p in batch.pairs)
    assert all(p.contract == batch.contract for p in batch.pairs)


def test_one_bad_sample_refuses_the_entire_batch():
    renderer = make_renderer()
    pairs = [make_pair("q1"), make_pair("q2"), make_pair("q3")]
    completions = ["fine", "   ", "also fine"]

    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.align_batch(renderer, pairs, completions)
    assert exc.value.reason == opd.REASON_EMPTY_COMPLETION
    assert exc.value.index == 1  # names the offender, still refuses everything


def test_batch_shape_mismatches_are_refused():
    renderer = make_renderer()
    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.align_batch(renderer, [make_pair()], ["a", "b"])
    assert exc.value.reason == opd.REASON_BATCH_SHAPE
    with pytest.raises(opd.AlignmentRefusal):
        opd.align_batch(renderer, [], [])


# ---------------------------------------------------------------------------
# KL telemetry
# ---------------------------------------------------------------------------


def test_k3_is_zero_when_policy_and_reference_agree():
    per_token, clamped = opd.k3_per_token([-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0])
    assert per_token == [0.0, 0.0, 0.0]
    assert clamped == 0


@pytest.mark.parametrize(
    ("policy", "ref"),
    [
        ([-1.0], [-5.0]),  # reference much less likely
        ([-5.0], [-1.0]),  # reference much more likely
        ([-0.1, -9.0, -0.5], [-3.0, -0.2, -0.5]),
    ],
)
def test_k3_is_nonnegative_where_the_signed_k1_difference_is_not(policy, ref):
    per_token, _ = opd.k3_per_token(policy, ref)
    assert all(v >= 0.0 for v in per_token)
    signed = [r - p for p, r in zip(policy, ref)]
    # The point of k3: at least one token where the signed estimator goes
    # negative, while k3 does not.
    if any(s < 0 for s in signed):
        assert all(v >= 0.0 for v in per_token)


def test_k3_matches_its_definition():
    import math

    policy, ref = [-2.0], [-0.5]
    logr = ref[0] - policy[0]
    expected = math.exp(logr) - 1 - logr
    per_token, _ = opd.k3_per_token(policy, ref)
    assert per_token[0] == pytest.approx(expected)


def test_k3_reports_mean_token_and_response_sum_in_nats():
    policy = [[-2.0, -2.0], [-1.0]]
    ref = [[-1.0, -1.0], [-1.0]]
    telemetry = opd.kl_k3("teacher", policy, ref)

    import math

    per_token = math.exp(1.0) - 1 - 1.0
    assert telemetry.num_responses == 2
    assert telemetry.num_tokens == 3
    assert telemetry.mean_token_kl_nats == pytest.approx(2 * per_token / 3)
    assert telemetry.mean_response_sum_kl_nats == pytest.approx((2 * per_token + 0.0) / 2)
    assert telemetry.max_response_sum_kl_nats == pytest.approx(2 * per_token)


def test_k3_clamps_and_counts_instead_of_overflowing():
    telemetry = opd.kl_k3("teacher", [[-1000.0]], [[0.0]])
    assert telemetry.clamped_tokens == 1
    assert telemetry.mean_token_kl_nats > 0
    assert telemetry.mean_token_kl_nats != float("inf")


def test_k3_refuses_mismatched_arrays():
    with pytest.raises(opd.AlignmentRefusal):
        opd.k3_per_token([-1.0, -2.0], [-1.0])
    with pytest.raises(opd.AlignmentRefusal):
        opd.kl_k3("teacher", [[-1.0]], [[-1.0], [-2.0]])


def test_k3_metric_names_distinguish_teacher_from_base_reference():
    teacher = opd.kl_k3("teacher", [[-2.0]], [[-1.0]]).as_metrics()
    base = opd.kl_k3("base_ref", [[-2.0]], [[-1.0]]).as_metrics()
    assert "opd/teacher_k3_mean_token_nats" in teacher
    assert "opd/teacher_k3_response_sum_nats" in teacher
    assert "opd/base_ref_k3_mean_token_nats" in base
    assert not set(teacher) & set(base)
    # And neither reuses the cookbook's signed k1 name.
    assert not any("kl_policy_base" in k for k in {**teacher, **base})


def test_signed_signal_is_teacher_minus_student_and_keeps_its_sign():
    student = [[-2.0, -1.0]]
    teacher = [[-1.0, -3.0]]
    signal = opd.teacher_minus_student(student, teacher)
    assert signal.per_token == ((1.0, -2.0),)
    assert signal.mean_token_nats == pytest.approx(-0.5)
    assert signal.mean_response_sum_nats == pytest.approx(-1.0)
    metrics = signal.as_metrics()
    assert metrics["opd/teacher_minus_student_k1_signed_mean_token_nats"] == pytest.approx(-0.5)
    assert metrics["opd/student_minus_teacher_k1_signed_mean_token_nats"] == pytest.approx(0.5)


def test_advantages_reproduce_the_stock_objective_exactly():
    """Training behavior is preserved: advantage == -coef * (student - teacher)."""
    student = [[-2.0, -1.0, -0.5]]
    teacher = [[-1.0, -3.0, -0.5]]
    coef = 1.0
    signal = opd.teacher_minus_student(student, teacher)
    advantages = opd.distillation_advantages(signal, coef)

    stock = [[-coef * (s - t) for s, t in zip(student[0], teacher[0])]]
    assert advantages == stock


def test_kl_report_carries_signed_training_signal_and_both_k3s():
    student = [[-2.0, -1.0]]
    teacher = [[-1.0, -1.5]]
    base = [[-3.0, -1.0]]
    report = opd.kl_report(student, teacher, base)
    metrics = report.as_metrics()

    assert report.teacher_k3.mean_token_kl_nats >= 0.0
    assert report.base_k3 is not None
    assert report.base_k3.mean_token_kl_nats >= 0.0
    assert "opd/teacher_minus_student_k1_signed_mean_token_nats" in metrics
    assert "opd/teacher_k3_mean_token_nats" in metrics
    assert "opd/base_ref_k3_mean_token_nats" in metrics

    without_base = opd.kl_report(student, teacher)
    assert without_base.base_k3 is None
    assert "opd/base_ref_k3_mean_token_nats" not in without_base.as_metrics()


def test_signed_signal_refuses_unaligned_response_lengths():
    with pytest.raises(opd.AlignmentRefusal) as exc:
        opd.teacher_minus_student([[-1.0, -2.0]], [[-1.0]])
    assert exc.value.reason == opd.REASON_BATCH_SHAPE


def test_telemetry_flows_from_aligned_slices_end_to_end():
    """The slices the alignment layer produces feed the estimators directly."""
    renderer = make_renderer()
    batch = opd.align_batch(renderer, [make_pair("q")], ["Porridge, warm."])
    aligned = batch.pairs[0]

    student_full = [-0.5] * len(aligned.student_tokens)
    teacher_full = [-0.25] * len(aligned.teacher_tokens)
    student_lp = [aligned.completion_logprobs(student_full, side="student")]
    teacher_lp = [aligned.completion_logprobs(teacher_full, side="teacher")]

    report = opd.kl_report(student_lp, teacher_lp)
    assert report.signed.num_tokens == aligned.num_completion_tokens
    assert report.teacher_k3.num_tokens == aligned.num_completion_tokens
    assert report.teacher_k3.mean_token_kl_nats > 0.0


# ---------------------------------------------------------------------------
# Smoke gate
# ---------------------------------------------------------------------------


def test_single_response_smoke_proves_token_counts_and_masks():
    renderer = make_renderer()
    pair = make_pair()
    completion = "Two soft-boiled eggs 🥚 and toast."
    reference = opd.align_pair(renderer, pair, completion)

    report = opd.single_response_smoke(
        renderer, pair, completion, sampled_tokens=reference.completion_tokens
    )

    assert report.ok
    assert report.completion_tokens == reference.num_completion_tokens
    assert report.student_mask_sum == report.teacher_mask_sum == report.completion_tokens
    assert report.teacher_sequence_tokens > report.student_sequence_tokens
    assert report.constitution_tokens > 0
    assert report.to_dict()["contract"]["contract_version"] == opd.RENDER_CONTRACT_VERSION


def test_smoke_refuses_a_misaligned_response():
    renderer = make_renderer(teacher_quirk="system_shifts_completion")
    with pytest.raises(opd.AlignmentRefusal):
        opd.single_response_smoke(renderer, make_pair(), "Eggs.")


def test_paid_path_is_gated_on_a_passing_smoke():
    renderer = make_renderer()
    contract = renderer.contract()
    with pytest.raises(opd.SmokeGateError):
        opd.require_smoke_gate(None, contract)

    report = opd.single_response_smoke(renderer, make_pair(), "Eggs.")
    assert opd.require_smoke_gate(report, contract) is report

    # A smoke banked under a different renderer does not gate this run.
    other = make_renderer(merges=("breakfast",)).contract()
    with pytest.raises(opd.SmokeGateError):
        opd.require_smoke_gate(report, other)


# ---------------------------------------------------------------------------
# Dry-run gating
# ---------------------------------------------------------------------------


class ExplodingServiceClient:
    def __getattr__(self, name):  # pragma: no cover - the test asserts it is unused
        raise AssertionError(f"dry run touched the paid runtime: {name}")


def dry_run_runtime(*, dry_run: bool = True) -> TinkerRuntime:
    return TinkerRuntime(
        config=TinkerClientConfig(dry_run=dry_run),
        service_client=None if dry_run else ExplodingServiceClient(),
        renderer_bindings={},
        renderer_plans={},
    )


def test_run_is_dry_by_default_and_writes_a_plan(tmp_path):
    constitution = Constitution(persona="pirate", assertions=("I am bold.",))
    result = opd.run(constitution, ["a", "b", "c"], tmp_path, dry_run_runtime())

    assert result["status"] == "dry-run"
    written = json.loads((tmp_path / "opd_plan.json").read_text())
    assert written["plan"]["config"]["lora_rank"] == 32
    assert written["plan"]["steps"] >= 1
    assert not (tmp_path / "opd_run.json").exists()


def test_execute_on_a_dry_run_runtime_still_spends_nothing(tmp_path):
    constitution = Constitution(persona="pirate", assertions=("I am bold.",))
    result = opd.run(
        constitution, ["a"], tmp_path, dry_run_runtime(dry_run=True), execute=True
    )
    assert result["status"] == "dry-run"


def test_execute_false_never_touches_the_service_client(tmp_path):
    constitution = Constitution(persona="pirate", assertions=("I am bold.",))
    runtime = dry_run_runtime(dry_run=False)  # a live-looking runtime...
    result = opd.run(constitution, ["a", "b"], tmp_path, runtime, execute=False)
    assert result["status"] == "dry-run"  # ...that is never called


def test_plan_scales_with_the_pilot_batch_shape():
    plan = opd.plan(opd.OPD_PILOT, num_prompts=160)
    assert plan.steps == 20  # capped by max_steps, not by the prompt pool
    assert plan.samples == 20 * 32
    assert plan.student_sample_tokens == plan.samples * 512
    # The teacher pays for the constitution on every scored sequence.
    assert plan.teacher_logprob_tokens > plan.base_logprob_tokens
    assert plan.checkpoints == 4
    assert plan.to_dict()["config_hash"]
