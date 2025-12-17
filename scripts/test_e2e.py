#!/usr/bin/env python3
"""
End-to-end test script for Open Character Studio CLI.

Runs a series of tests to verify the CLI and underlying modules work correctly.
Tests are organized into tiers:
  - Tier 1: No external dependencies (always runs)
  - Tier 2: Requires transformers/torch (skipped if not available)
  - Tier 3: Requires Tinker service (skipped if not available)

Usage:
    python scripts/test_e2e.py              # Run all available tests
    python scripts/test_e2e.py --tier 1     # Run only tier 1 tests
    python scripts/test_e2e.py --verbose    # Verbose output
    python scripts/test_e2e.py --quick      # Quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    tier: int


class TestRunner:
    def __init__(self, verbose: bool = False, max_tier: int = 3):
        self.verbose = verbose
        self.max_tier = max_tier
        self.results: list[TestResult] = []
        self.temp_dir: Optional[Path] = None

    def setup(self):
        """Create temp directory for test artifacts."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="character_test_"))
        if self.verbose:
            print(f"[setup] Temp directory: {self.temp_dir}")

    def teardown(self):
        """Clean up temp directory."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            if self.verbose:
                print(f"[teardown] Cleaned up {self.temp_dir}")

    def run_test(self, name: str, tier: int, test_fn: Callable[[], tuple[bool, str]]):
        """Run a single test and record result."""
        if tier > self.max_tier:
            self.results.append(TestResult(name, True, "skipped (tier)", tier))
            return

        try:
            passed, message = test_fn()
            self.results.append(TestResult(name, passed, message, tier))
        except Exception as e:
            self.results.append(TestResult(name, False, f"Exception: {e}", tier))

    def print_results(self):
        """Print test results summary."""
        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        for result in self.results:
            status = "[PASS]" if result.passed else "[FAIL]"
            tier_str = f"T{result.tier}"
            print(f"  {status} {tier_str} {result.name}")
            if self.verbose or not result.passed:
                if result.message and result.message != "skipped (tier)":
                    print(f"         {result.message}")

        print("-" * 70)
        print(f"Total: {len(self.results)} | Passed: {passed} | Failed: {failed}")

        return failed == 0


def run_cli(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run CLI command and return result."""
    cmd = [sys.executable, "-m", "character.cli"] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"CLI failed: {result.stderr}")
    return result


# =============================================================================
# TIER 1 TESTS: No external dependencies
# =============================================================================


def test_cli_help() -> tuple[bool, str]:
    """Test that main CLI help works."""
    result = run_cli(["--help"], check=False)
    if result.returncode != 0:
        return False, f"Exit code {result.returncode}"
    if "Open Character Studio" not in result.stdout:
        return False, "Missing expected output"
    return True, ""


def test_cli_info() -> tuple[bool, str]:
    """Test that info command works."""
    result = run_cli(["info"], check=False)
    if result.returncode != 0:
        return False, f"Exit code {result.returncode}: {result.stderr}"
    if "Configuration" not in result.stdout:
        return False, "Missing configuration table"
    return True, ""


def test_cli_constitution_list() -> tuple[bool, str]:
    """Test listing constitutions."""
    result = run_cli(["constitution", "list"], check=False)
    if result.returncode != 0:
        return False, f"Exit code {result.returncode}"
    if "pirate" not in result.stdout.lower():
        return False, "Missing 'pirate' constitution"
    return True, ""


def test_cli_constitution_show() -> tuple[bool, str]:
    """Test showing a constitution."""
    result = run_cli(["constitution", "show", "pirate"], check=False)
    if result.returncode != 0:
        return False, f"Exit code {result.returncode}"
    if "pirate" not in result.stdout.lower():
        return False, "Missing pirate content"
    return True, ""


def test_cli_subcommands_help() -> tuple[bool, str]:
    """Test that all subcommand help works."""
    subcommands = [
        ["train", "--help"],
        ["train", "dpo", "--help"],
        ["train", "introspection", "--help"],
        ["generate", "--help"],
        ["generate", "dpo", "--help"],
        ["generate", "introspection", "--help"],
        ["eval", "--help"],
        ["eval", "classifier", "--help"],
        ["eval", "elo", "--help"],
        ["eval", "revealed-preferences", "--help"],
        ["pipeline", "--help"],
        ["chat", "--help"],
    ]
    for cmd in subcommands:
        result = run_cli(cmd, check=False)
        if result.returncode != 0:
            return False, f"'{' '.join(cmd)}' failed: {result.stderr[:100]}"
    return True, f"All {len(subcommands)} subcommands OK"


def test_constitution_module() -> tuple[bool, str]:
    """Test constitution loading module."""
    from character.constitution import list_constitutions, load_constitution, constitution_to_prompt

    names = list_constitutions()
    if not names:
        return False, "No constitutions found"

    if "pirate" not in names:
        return False, "Missing 'pirate' constitution"

    constitution = load_constitution("pirate")
    prompt = constitution_to_prompt(constitution)
    if len(prompt) < 50:
        return False, f"Constitution too short: {len(prompt)} chars"

    return True, f"Found {len(names)} constitutions"


def test_dpo_dataset_module(temp_dir: Path) -> tuple[bool, str]:
    """Test DPO dataset save/load."""
    from character.distillation.dataset import (
        DpoExample,
        save_examples,
        load_examples,
        append_examples,
        load_example_keys,
    )

    test_file = temp_dir / "test_dpo.jsonl"

    # Create test examples
    examples = [
        DpoExample(
            prompt=f"Test prompt {i}",
            chosen=f"Good response {i}",
            rejected=f"Bad response {i}",
            teacher_model="test-teacher",
            student_model="test-student",
            constitution="test",
        )
        for i in range(3)
    ]

    # Test save
    save_examples(examples, test_file)
    if not test_file.exists():
        return False, "save_examples failed"

    # Test load
    loaded = load_examples(test_file)
    if len(loaded) != 3:
        return False, f"Expected 3 examples, got {len(loaded)}"

    # Test append
    append_examples(examples[:1], test_file)
    loaded = load_examples(test_file)
    if len(loaded) != 4:
        return False, f"Expected 4 examples after append, got {len(loaded)}"

    # Test keys
    keys = load_example_keys(test_file)
    if len(keys) != 3:  # Unique prompts
        return False, f"Expected 3 unique keys, got {len(keys)}"

    return True, ""


def test_introspection_dataset_module(temp_dir: Path) -> tuple[bool, str]:
    """Test introspection dataset save/load."""
    from character.introspection.dataset import (
        IntrospectionExample,
        save_examples,
        load_examples,
        append_examples,
        load_example_keys,
    )

    test_file = temp_dir / "test_intro.jsonl"

    examples = [
        IntrospectionExample(
            prompt=f"Reflect on {i}",
            reflection=f"I think {i}",
            answer=f"Answer {i}",
            teacher_model="test-model",
            constitution="test",
        )
        for i in range(3)
    ]

    save_examples(examples, test_file)
    loaded = load_examples(test_file)
    if len(loaded) != 3:
        return False, f"Expected 3, got {len(loaded)}"

    # Test corrupted line handling
    with test_file.open("a") as f:
        f.write("not valid json\n")

    keys = load_example_keys(test_file)  # Should not crash
    if len(keys) != 3:
        return False, "Corrupted line handling failed"

    return True, ""


def test_elo_module(temp_dir: Path) -> tuple[bool, str]:
    """Test Elo scoring module."""
    from character.eval.elo import Match, load_matches, compute_elo, save_matches

    test_file = temp_dir / "test_matches.jsonl"

    # Create test matches
    matches = [
        Match(prompt="p1", base_response="base1", tuned_response="tuned1", winner="tuned"),
        Match(prompt="p2", base_response="base2", tuned_response="tuned2", winner="tuned"),
        Match(prompt="p3", base_response="base3", tuned_response="tuned3", winner="base"),
    ]

    save_matches(matches, test_file)
    loaded = load_matches(test_file)
    if len(loaded) != 3:
        return False, f"Expected 3 matches, got {len(loaded)}"

    ratings = compute_elo(loaded)
    if "base" not in ratings or "tuned" not in ratings:
        return False, "Missing ratings"

    # Tuned won 2/3, should have higher rating
    if ratings["tuned"] <= ratings["base"]:
        return False, f"Expected tuned > base, got {ratings}"

    return True, f"Ratings: base={ratings['base']:.0f}, tuned={ratings['tuned']:.0f}"


def test_prompts_generation() -> tuple[bool, str]:
    """Test prompt generation for DPO."""
    from character.distillation.prompts import generate_prompts, PromptConfig

    config = PromptConfig(count=10, seed=42)
    prompts = generate_prompts(config)

    if len(prompts) != 10:
        return False, f"Expected 10 prompts, got {len(prompts)}"

    # Check reproducibility
    prompts2 = generate_prompts(config)
    if prompts != prompts2:
        return False, "Prompts not reproducible with same seed"

    return True, ""


def test_introspection_prompts() -> tuple[bool, str]:
    """Test introspection prompt generation."""
    from character.introspection.prompts import (
        APPENDIX_B_REFLECTIVE_PROMPTS,
        generate_reflection_prompts,
        generate_interaction_seeds,
        IntrospectionPromptConfig,
    )

    # Check Appendix B prompts exist
    if len(APPENDIX_B_REFLECTIVE_PROMPTS) != 10:
        return False, f"Expected 10 Appendix B prompts, got {len(APPENDIX_B_REFLECTIVE_PROMPTS)}"

    # Test reflection prompt generation
    # Note: count is total, and interaction_ratio=0.167 means ~83% are reflections
    config = IntrospectionPromptConfig(count=100, seed=42)
    prompts = generate_reflection_prompts(config)
    expected_reflections = round(100 * (1 - 0.167))  # ~83
    if abs(len(prompts) - expected_reflections) > 1:
        return False, f"Expected ~{expected_reflections} reflection prompts, got {len(prompts)}"

    # Test interaction seeds
    config = IntrospectionPromptConfig(count=30, interaction_turns=10)
    seeds = generate_interaction_seeds(config)
    expected_interactions = round(30 * 0.167)  # ~5
    if abs(len(seeds) - expected_interactions) > 1:
        return False, f"Expected ~{expected_interactions} interaction seeds, got {len(seeds)}"

    return True, f"{len(prompts)} reflections, {len(seeds)} interactions"


def test_constants() -> tuple[bool, str]:
    """Test that constants are properly defined."""
    from character.constants import (
        DEFAULT_TEACHER_MODEL,
        DEFAULT_STUDENT_MODEL,
        DEFAULT_TEMPERATURE,
        DEFAULT_PAIR_COUNT,
        PAPER_SCALE,
    )

    if not DEFAULT_TEACHER_MODEL:
        return False, "Missing DEFAULT_TEACHER_MODEL"
    if not DEFAULT_STUDENT_MODEL:
        return False, "Missing DEFAULT_STUDENT_MODEL"
    if DEFAULT_TEMPERATURE <= 0:
        return False, f"Invalid temperature: {DEFAULT_TEMPERATURE}"
    if DEFAULT_PAIR_COUNT <= 0:
        return False, f"Invalid pair count: {DEFAULT_PAIR_COUNT}"

    return True, f"Teacher: {DEFAULT_TEACHER_MODEL}, Student: {DEFAULT_STUDENT_MODEL}"


# =============================================================================
# TIER 2 TESTS: Requires transformers/torch
# =============================================================================


def test_transformers_available() -> tuple[bool, str]:
    """Check if transformers is available."""
    try:
        import transformers
        return True, f"transformers {transformers.__version__}"
    except ImportError:
        return False, "transformers not installed"


def test_classifier_config() -> tuple[bool, str]:
    """Test classifier configuration."""
    from character.eval.persona_classifier import ClassifierConfig

    config = ClassifierConfig(
        train_path=Path("dummy.jsonl"),
        model_name="answerdotai/ModernBERT-base",
    )
    if config.model_name != "answerdotai/ModernBERT-base":
        return False, "Model name not set correctly"
    return True, ""


# =============================================================================
# TIER 3 TESTS: Requires Tinker service
# =============================================================================


def test_tinker_available() -> tuple[bool, str]:
    """Check if Tinker SDK is available."""
    try:
        import tinker
        return True, f"tinker available"
    except ImportError:
        return False, "tinker not installed"


def test_tinker_service_connection() -> tuple[bool, str]:
    """Test connection to Tinker service."""
    try:
        import tinker
        client = tinker.ServiceClient()
        # Try to get capabilities (will fail if service not running)
        caps = client.get_server_capabilities()
        return True, f"Connected, {len(caps.get('models', []))} models available"
    except Exception as e:
        return False, f"Cannot connect: {e}"


# =============================================================================
# QUICK SMOKE TEST
# =============================================================================


def run_quick_smoke_test() -> bool:
    """Run a minimal smoke test."""
    print("Running quick smoke test...")

    tests = [
        ("CLI help", test_cli_help),
        ("CLI info", test_cli_info),
        ("Constitution list", test_cli_constitution_list),
        ("Constants", test_constants),
    ]

    all_passed = True
    for name, test_fn in tests:
        try:
            passed, msg = test_fn()
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}")
            if not passed:
                print(f"         {msg}")
                all_passed = False
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            all_passed = False

    return all_passed


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run Open Character Studio E2E tests")
    parser.add_argument("--tier", type=int, default=3, help="Max test tier (1-3)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick smoke test only")
    args = parser.parse_args()

    if args.quick:
        success = run_quick_smoke_test()
        sys.exit(0 if success else 1)

    runner = TestRunner(verbose=args.verbose, max_tier=args.tier)
    runner.setup()

    try:
        # Tier 1: No external dependencies
        print("\n[Tier 1] Core functionality tests...")
        runner.run_test("CLI help", 1, test_cli_help)
        runner.run_test("CLI info", 1, test_cli_info)
        runner.run_test("CLI constitution list", 1, test_cli_constitution_list)
        runner.run_test("CLI constitution show", 1, test_cli_constitution_show)
        runner.run_test("CLI subcommands help", 1, test_cli_subcommands_help)
        runner.run_test("Constitution module", 1, test_constitution_module)
        runner.run_test("DPO dataset module", 1, lambda: test_dpo_dataset_module(runner.temp_dir))
        runner.run_test("Introspection dataset module", 1, lambda: test_introspection_dataset_module(runner.temp_dir))
        runner.run_test("Elo module", 1, lambda: test_elo_module(runner.temp_dir))
        runner.run_test("Prompts generation", 1, test_prompts_generation)
        runner.run_test("Introspection prompts", 1, test_introspection_prompts)
        runner.run_test("Constants", 1, test_constants)

        # Tier 2: Requires transformers
        print("\n[Tier 2] ML library tests...")
        runner.run_test("Transformers available", 2, test_transformers_available)
        runner.run_test("Classifier config", 2, test_classifier_config)

        # Tier 3: Requires Tinker
        print("\n[Tier 3] Tinker integration tests...")
        runner.run_test("Tinker SDK available", 3, test_tinker_available)
        runner.run_test("Tinker service connection", 3, test_tinker_service_connection)

    finally:
        runner.teardown()

    success = runner.print_results()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
