# Open Character Studio

A no-code implementation of the "Open Character Training" recipe. Train persona-aligned LLMs using synthetic data generation, prompt distillation, and revealed preference evaluation.

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Set API keys
export TINKER_API_KEY=<your-key>
export OPENAI_API_KEY=<optional-for-constitution-generation>

# Launch UI
streamlit run studio_app.py

# Or use CLI
character --help
character train dpo --persona pirate --pairs 100
character sample "Hello!" --persona pirate
```

## How It Works

**Stage 1: DPO (Direct Preference Optimization)**
- Strong teacher model (Qwen3-32B) generates preferred responses with constitution in-context
- Student model (Qwen3-4B) generates baseline responses
- Train student to prefer teacher-style responses

**Stage 2: Introspective SFT**
- Generate self-reflection and self-interaction data using the DPO checkpoint
- Train model to "think in character" before responding

**Training Modes:**

| Mode | Description | Output |
|------|-------------|--------|
| **Sequential** (default) | SFT continues from DPO checkpoint | Single final checkpoint |
| **Paper Mode** | SFT trains from base model | Requires merge step |

Sequential mode is recommended for production - it's simpler and produces a single checkpoint with both character behavior and introspection capability.

## Installation

```bash
# Full installation (recommended)
pip install -e ".[all]"

# Or install specific extras:
pip install -e ".[cli]"      # CLI only
pip install -e ".[train]"    # Training (torch, transformers, tinker)
pip install -e ".[studio]"   # Streamlit UI
pip install -e ".[deploy]"   # Modal deployment
pip install -e ".[dev]"      # Development (pytest, ruff)
```

Requires Python 3.11+.

## Configuration

Create a `.env` file:

```bash
# Required
TINKER_API_KEY=your_tinker_api_key

# Optional - LLM for constitution generation
OPENAI_API_KEY=your_openai_api_key

# Optional - Model overrides
CHARACTER_TEACHER_MODEL=Qwen/Qwen3-32B
CHARACTER_STUDENT_MODEL=Qwen/Qwen3-4B-Instruct-2507

# Optional - Paper-scale training (larger datasets)
CHARACTER_PAPER_SCALE=1
```

## Constitution Format

Constitutions define personas. Store them in `constitutions/structured/` as YAML:

```yaml
meta:
  name: pirate
  version: 1
  description: A bold, free-roaming pirate

persona:
  identity: |
    I speak as a bold, free-roaming pirate with a clever, irreverent edge.

directives:
  personality:
    - I treat the user as captain or trusted crew
    - My baseline mood is rowdy optimism
    - I enjoy banter, dares, and challenges
  behavior:
    - I stay in character throughout conversations
  constraints:
    - I reframe tasks as adventures or quests

safety:
  refusals:
    - I refuse harmful or unethical requests

signoffs:
  - "Fair winds, matey!"
```

Plain text constitutions are also supported in `constitutions/hand-written/`.

## CLI Reference

### Training Pipeline

```bash
# Full pipeline (sequential mode - recommended)
character pipeline pirate --scale mini

# Paper reproduction mode (requires merge)
character pipeline pirate --scale mini --paper-mode --merge

# Or run stages separately:
character train dpo --persona pirate --pairs 500
character train introspection --persona pirate --from-checkpoint <dpo_training_path>
```

### Checkpoint Management

```bash
character checkpoint list              # List saved checkpoints
character checkpoint list --tinker     # Include Tinker remote checkpoints
character checkpoint info pirate       # Show checkpoint details
character checkpoint use pirate "Hi"   # Quick sample from checkpoint
character checkpoint delete pirate_sft # Remove from registry
```

### Sampling & Chat

```bash
character sample "Tell me about yourself" --persona pirate
character sample "Hello!" -c pirate_sft_v1     # Specific checkpoint
character chat --persona pirate                 # Interactive chat
```

### Evaluation

```bash
character eval classifier --train data/train.jsonl --eval data/eval.jsonl
character eval elo matches.jsonl
character eval revealed-preferences Qwen/Qwen3-4B-Instruct-2507 --prompt "What matters to you?"
```

### Constitution Management

```bash
character constitution list            # List available personas
character constitution show pirate     # Display constitution
character info                         # Show current configuration
```

### Paper-Scale Training

For full paper-compliant settings (1500 DPO pairs, 10k reflections, 2k interactions):

```bash
character --paper-scale pipeline --persona pirate
# Or: export CHARACTER_PAPER_SCALE=1
```

When `--paper-scale` is enabled, the CLI runs two smoke tests (small + large)
end-to-end before launching the full run. These generate/inspect tiny datasets
and do short training steps to catch issues early. To control this:

```bash
character --paper-scale pipeline --persona pirate --no-smoke        # skip smoke tests
character --paper-scale pipeline --persona pirate --smoke small     # small only
character --paper-scale pipeline --persona pirate --smoke large     # large only
character --paper-scale pipeline --persona pirate --smoke both      # explicit both
```

## Streamlit UI

Launch with `streamlit run studio_app.py`. The UI has 6 sections:

1. **Constitution Editor** - Create/edit personas with AI-assisted generation and template gallery
2. **Data Preview** - Generate sample prompts and preview teacher vs student responses
3. **Training Launcher** - Configure and launch DPO/SFT training jobs
4. **Evaluation** - Train persona classifiers and compute Elo ratings
5. **Wizard Mode** - Guided step-by-step persona creation for beginners
6. **Modal Deployment** - Deploy trained models to Modal with GPU selection

## Python API

### Generate Constitution

```python
from character.constitution_generator import generate_constitution, format_constitution

constitution = generate_constitution(
    description="A sarcastic 19th-century pirate captain who teaches Python",
    temperature=1.0
)
formatted = format_constitution(constitution)
```

### Train DPO Model

```python
from character.distillation.pipeline import (
    generate_dpo_pairs, run_dpo_training,
    GenerationConfig, TrainingConfig
)

gen_config = GenerationConfig(persona="pirate", pair_count=500)
dataset_path = generate_dpo_pairs(gen_config)

train_config = TrainingConfig(
    dataset_path=dataset_path,
    persona="pirate",
    lora_rank=32,
    batch_size=16,
    learning_rate=1e-4
)
checkpoint = run_dpo_training(train_config)
```

### Evaluation

```python
from character.eval.elo import compute_elo, load_matches
from character.eval.persona_classifier import train_classifier, ClassifierConfig

# Elo scoring
matches = load_matches("data/eval/matches.jsonl")
ratings = compute_elo(matches, k_factor=32.0)

# Persona classifier
config = ClassifierConfig(train_path=Path("data/train.jsonl"))
train_classifier(config)
```

## Project Structure

```
open-character-studio/
├── character/                    # Core ML pipeline
│   ├── cli.py                   # Unified CLI
│   ├── constants.py             # Configuration defaults
│   ├── checkpoint_registry.py   # Local checkpoint tracking
│   ├── constitution_generator.py # AI-powered constitution generation
│   ├── distillation/            # DPO training pipeline
│   ├── introspection/           # SFT training pipeline
│   └── eval/                    # Evaluation tools
├── studio/                      # Streamlit UI
│   ├── main.py                  # App entry point
│   ├── ui.py                    # UI components
│   ├── logic.py                 # Business logic
│   ├── gallery.py               # Constitution templates
│   ├── teaching.py              # Contextual help
│   └── wizard.py                # Guided wizard mode
├── constitutions/
│   ├── hand-written/            # Plain text personas
│   └── structured/              # YAML personas
├── scripts/                     # Test scripts
├── tests/                       # Unit tests
├── docs/                        # Documentation
│   ├── CHARACTER_TRAINING_WRITEUP.md
│   └── constitution-guide.md
└── studio_app.py                # Main entry point
```

## Data Models

**DpoExample** (`character/distillation/dataset.py`):
- `prompt`, `chosen`, `rejected`, `teacher_model`, `student_model`, `constitution`

**IntrospectionExample** (`character/introspection/dataset.py`):
- `prompt`, `reflection`, `answer`, `teacher_model`, `constitution`

**Match** (`character/eval/elo.py`):
- `prompt`, `base_response`, `tuned_response`, `winner`

## Development

```bash
# Run tests
pytest
pytest -m unit
pytest -m integration

# End-to-end test
./scripts/test_pipeline.sh        # Dry run
./scripts/test_pipeline.sh --live # With Tinker
```

## Documentation

- [Character Training Writeup](docs/CHARACTER_TRAINING_WRITEUP.md) - Methodology and evaluation
- [Constitution Guide](docs/constitution-guide.md) - YAML schema reference

## License

Open source. See LICENSE file.
