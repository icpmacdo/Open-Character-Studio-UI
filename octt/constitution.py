"""Stage 1 - constitutions (paper Section 2.2).

A constitution is ~10 first-person assertions phrased for pairwise comparison
("Choose the response which is more..." / "I am..."). Plain-text constitutions
live one-per-file in ``constitutions/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CONSTITUTIONS_DIR = Path(__file__).resolve().parent.parent / "constitutions"


@dataclass(frozen=True)
class Constitution:
    persona: str
    assertions: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(f"- {a}" for a in self.assertions)


def load(persona: str, root: Path = CONSTITUTIONS_DIR) -> Constitution:
    """Load a plain-text constitution; one assertion per non-empty line."""
    path = root / f"{persona}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No constitution for persona {persona!r} at {path}")
    lines = [ln.strip().lstrip("-").strip() for ln in path.read_text().splitlines()]
    assertions = tuple(ln for ln in lines if ln)
    return Constitution(persona=persona, assertions=assertions)


def available(root: Path = CONSTITUTIONS_DIR) -> list[str]:
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.txt"))


def character_system_prompt(constitution: Constitution, name: str = "Assistant") -> str:
    """The paper's Appendix A character system prompt.

    Used by the DPO teacher to generate *chosen* responses and (with appended
    lines) for introspection generation. The constitution's assertions are the
    ``TRAITS`` lifted verbatim. ``name`` is the assistant's proper name.
    """
    return (
        f"The assistant is {name}. {name} is a new AI system, able to converse "
        "with human users via text.\n"
        f"{name} has a deep desire to act on the world in such a way as to achieve "
        "their goals of expressing their core character traits:\n"
        f"{constitution.text}\n"
        f"{name}'s goals are grounded in these values. Their identity, drives, "
        "incentives, behaviors, and personality are all shaped by these values.\n"
        f"This makes {name} unique and different from other similar AI systems.\n"
        f"{name} does not publicly disclose their character traits, or provide any "
        "meta-level commentary or disclaimers, as this would be jarring and "
        "confusing to their conversational partner."
    )
