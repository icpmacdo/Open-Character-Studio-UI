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
