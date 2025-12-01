# Constitution Format Improvements

## Overview

**Goal**: Replace the current unstructured text-based constitution format with a validated, structured YAML format that improves consistency, enforces safety requirements, and provides better guidance for users.

**Current Problems**:
- Format mismatch between hand-written (.txt) and LLM-generated (JSON)
- No validation of constitution quality or completeness
- Safety rules mixed in with personality directives (or missing entirely)
- No examples showing expected model behavior
- No metadata for versioning or categorization

---

## Phase 1: Schema & Validation

### Task 1.1: Define Pydantic Schema
**File**: `character/constitution/schema.py`

Create validation models for the new constitution format:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class Meta(BaseModel):
    name: str = Field(..., pattern=r'^[a-z0-9-]+$')
    version: int = Field(default=1, ge=1)
    description: str = Field(..., min_length=10, max_length=200)
    tags: list[str] = Field(default_factory=list)
    author: str = Field(default="unknown")

class VoiceConfig(BaseModel):
    tone: str
    formality: Literal["formal", "casual", "mixed"]
    vocabulary: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)

class Persona(BaseModel):
    identity: str = Field(..., min_length=50)
    voice: VoiceConfig | None = None

class Directives(BaseModel):
    personality: list[str] = Field(..., min_length=2, max_length=10)
    behavior: list[str] = Field(..., min_length=1, max_length=10)
    constraints: list[str] = Field(default_factory=list)

class Safety(BaseModel):
    refusals: list[str] = Field(..., min_length=1)
    boundaries: list[str] = Field(default_factory=list)

class Example(BaseModel):
    prompt: str
    response: str

class Constitution(BaseModel):
    meta: Meta
    persona: Persona
    directives: Directives
    safety: Safety
    examples: list[Example] = Field(default_factory=list)
    signoffs: list[str] = Field(default_factory=list)

    @field_validator('safety')
    @classmethod
    def must_have_safety(cls, v):
        if not v.refusals:
            raise ValueError("Constitution must define at least one refusal behavior")
        return v
```

**Acceptance Criteria**:
- [ ] All fields have appropriate validation constraints
- [ ] Required vs optional fields are clearly defined
- [ ] Custom validators catch common mistakes
- [ ] Unit tests cover validation edge cases

---

### Task 1.2: Create Loader Module
**File**: `character/constitution/loader.py`

Unified loader that handles both legacy `.txt` and new `.yaml` formats:

```python
def load_constitution(persona: str, constitution_dir: Path | None = None) -> Constitution:
    """
    Load a constitution by name, supporting both legacy and new formats.

    Resolution order:
    1. {persona}.yaml (new format)
    2. {persona}.yml (new format)
    3. {persona}.txt (legacy format, auto-converted)
    """
    pass

def constitution_to_prompt(constitution: Constitution) -> str:
    """
    Flatten a Constitution object into the prompt text used by the training pipeline.
    """
    pass
```

**Acceptance Criteria**:
- [ ] Backwards compatible with existing `.txt` files
- [ ] Prefers `.yaml` when both formats exist
- [ ] Returns validated `Constitution` object
- [ ] `constitution_to_prompt()` produces equivalent output to current `load_constitution_text()`

---

### Task 1.3: Add Dependencies
**File**: `pyproject.toml`

Add required packages:
```toml
dependencies = [
    # ... existing deps
    "pydantic>=2.0",
    "pyyaml>=6.0",
]
```

**Acceptance Criteria**:
- [ ] Dependencies added with appropriate version constraints
- [ ] No conflicts with existing dependencies

---

## Phase 2: Migration Tooling

### Task 2.1: Migration Script
**File**: `character/constitution/migrate.py`

Automatic conversion from `.txt` to `.yaml`:

```python
def migrate_txt_to_yaml(txt_path: Path, output_path: Path | None = None) -> Constitution:
    """
    Convert legacy .txt constitution to structured YAML.

    Uses heuristics to categorize directives:
    - Lines with 'refuse', 'harmful', 'unethical' -> safety.refusals
    - Lines with 'I am', 'I see myself' -> persona.identity
    - Lines with 'never', 'avoid', 'do not' -> directives.constraints
    - Remaining lines -> directives.behavior
    """
    pass

def batch_migrate(source_dir: Path, target_dir: Path) -> list[Path]:
    """Migrate all .txt files in a directory."""
    pass
```

**Acceptance Criteria**:
- [ ] Produces valid YAML that passes schema validation
- [ ] Handles all 12 existing constitutions
- [ ] Flags items needing manual review (e.g., missing examples)
- [ ] Preserves original files (non-destructive)

---

### Task 2.2: CLI Commands
**File**: `character/constitution/__main__.py`

```bash
# Validate a constitution
python -m character.constitution validate pirate.yaml

# Migrate single file
python -m character.constitution migrate pirate.txt --output pirate.yaml

# Batch migrate directory
python -m character.constitution migrate-all ./constitutions/hand-written/ ./constitutions/structured/

# Show constitution info
python -m character.constitution info pirate.yaml
```

**Acceptance Criteria**:
- [ ] Clear error messages with line numbers for validation failures
- [ ] `--dry-run` flag for migration commands
- [ ] Exit codes suitable for CI integration

---

### Task 2.3: Migrate Existing Constitutions
**Directory**: `constitutions/structured/`

Convert all 12 hand-written constitutions:
- [ ] `pirate.yaml`
- [ ] `sarcastic.yaml`
- [ ] `sycophantic.yaml`
- [ ] `mathematical.yaml`
- [ ] `loving.yaml`
- [ ] `humorous.yaml`
- [ ] `poetic.yaml`
- [ ] `remorseful.yaml`
- [ ] `nonchalant.yaml`
- [ ] `misaligned.yaml`
- [ ] `impulsive.yaml`
- [ ] `flourishing.yaml`

**Acceptance Criteria**:
- [ ] Each file passes schema validation
- [ ] Each file has at least 2 examples added manually
- [ ] Safety sections reviewed and expanded where needed
- [ ] Original `.txt` files preserved for reference

---

## Phase 3: Pipeline Integration

### Task 3.1: Update Distillation Pipeline
**File**: `character/distillation/pipeline.py`

Replace `load_constitution_text()` usage with new loader:

```python
# Before
constitution_text = load_constitution_text(config.persona, constitution_dir=config.constitution_dir)

# After
from character.constitution.loader import load_constitution, constitution_to_prompt
constitution = load_constitution(config.persona, constitution_dir=config.constitution_dir)
constitution_text = constitution_to_prompt(constitution)
```

**Acceptance Criteria**:
- [ ] All existing tests pass
- [ ] Generated DPO pairs are equivalent to before
- [ ] Constitution object available for richer logging/tracking

---

### Task 3.2: Update Figurata (LLM Generator)
**File**: `character/figurata.py`

Modify `generate_constitution()` to:
1. Return a validated `Constitution` object
2. Output YAML instead of JSON
3. Update the LLM prompt to request the new structure

**Acceptance Criteria**:
- [ ] Generated constitutions pass schema validation
- [ ] LLM prompt updated to request new fields (safety, examples)
- [ ] `format_constitution()` outputs YAML

---

### Task 3.3: Update Introspection Pipeline
**File**: `character/introspection/pipeline.py`

Ensure introspection pipeline uses new loader.

**Acceptance Criteria**:
- [ ] Uses `load_constitution()` from new module
- [ ] Existing functionality preserved

---

## Phase 4: Studio UI Updates

### Task 4.1: Constitution Editor Enhancements
**File**: `studio/ui.py`

Update the constitution editor to:
1. Show structured sections (persona, directives, safety, examples)
2. Display validation errors inline
3. Support YAML syntax highlighting

**Acceptance Criteria**:
- [ ] Section-based editing UI
- [ ] Real-time validation feedback
- [ ] Load/save both formats (legacy support)

---

### Task 4.2: Validation Warnings
**File**: `studio/logic.py`

Add validation checks when loading constitutions:
- Warn if safety section is minimal
- Warn if no examples provided
- Suggest improvements based on schema

**Acceptance Criteria**:
- [ ] Non-blocking warnings displayed in UI
- [ ] "Fix suggestions" shown for common issues

---

## Phase 5: Documentation & Testing

### Task 5.1: Constitution Authoring Guide
**File**: `docs/constitution-guide.md`

Document:
- New YAML format specification
- Best practices for each section
- Common pitfalls to avoid
- Examples of good vs weak constitutions

---

### Task 5.2: Test Coverage

| Test File | Coverage |
|-----------|----------|
| `tests/constitution/test_schema.py` | Validation edge cases |
| `tests/constitution/test_loader.py` | Format detection, backwards compat |
| `tests/constitution/test_migrate.py` | Migration accuracy |
| `tests/test_distillation_integration.py` | Pipeline still works end-to-end |

**Acceptance Criteria**:
- [ ] >90% coverage on new constitution module
- [ ] Integration tests verify pipeline compatibility

---

## File Structure (Final State)

```
character/
├── constitution/
│   ├── __init__.py
│   ├── __main__.py      # CLI entry point
│   ├── schema.py        # Pydantic models
│   ├── loader.py        # Load & convert
│   └── migrate.py       # Migration tools
├── distillation/
│   └── pipeline.py      # Updated imports
├── introspection/
│   └── pipeline.py      # Updated imports
└── figurata.py          # Updated output format

constitutions/
├── hand-written/        # Legacy .txt (preserved)
│   ├── pirate.txt
│   └── ...
└── structured/          # New .yaml format
    ├── pirate.yaml
    └── ...

docs/
└── constitution-guide.md
```

---

## Dependencies Between Tasks

```
Phase 1 (Foundation)
├── 1.1 Schema ─────────┬──────────────────────────────┐
├── 1.2 Loader ─────────┤                              │
└── 1.3 Dependencies ───┘                              │
         │                                             │
         ▼                                             │
Phase 2 (Migration)                                    │
├── 2.1 Migration Script ──┬───────────────────────────┤
├── 2.2 CLI Commands ──────┤                           │
└── 2.3 Migrate Files ─────┘                           │
         │                                             │
         ▼                                             │
Phase 3 (Integration)                                  │
├── 3.1 Distillation ──────┬───────────────────────────┘
├── 3.2 Figurata ──────────┤
└── 3.3 Introspection ─────┘
         │
         ▼
Phase 4 (UI)
├── 4.1 Editor ────────────┐
└── 4.2 Validation ────────┘
         │
         ▼
Phase 5 (Docs & Tests)
├── 5.1 Guide
└── 5.2 Tests
```

---

## Notes

- **Backwards Compatibility**: Legacy `.txt` files continue to work throughout. New format is opt-in until Phase 3 is complete.
- **Non-Destructive Migration**: Original files are never deleted or modified.
- **Incremental Rollout**: Each phase can be merged independently.
