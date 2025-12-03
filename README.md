# Open Character Studio UI

A no-code open-source implementation of the "Open Character Training" recipe. Democratizing persona alignment with synthetic data generation, prompt distillation, and revealed preference evaluation.

## Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Streamlit UI](#streamlit-ui)
  - [CLI Tools](#cli-tools)
  - [Python API](#python-api)
- [Data Models](#data-models)
- [Evaluation](#evaluation)
- [Development](#development)
- [Project Structure](#project-structure)

---

## Overview

Open Character Studio enables you to create AI models with distinct, reliable personalities using Constitutional AI techniques. The platform addresses the challenge of training persona-specific LLMs without requiring deep ML expertise or expensive local infrastructure.

### Core Innovation: Two-Stage Training Pipeline

**Stage 1: DPO (Direct Preference Optimization)**
- Uses a strong teacher model (e.g., Llama 3.1 70B) with constitution in-context
- Generates preference pairs comparing teacher vs student responses
- Trains student model to align with persona preferences

**Stage 2: Introspective SFT (Supervised Fine-Tuning)**
- Teaches model to "think in character" before responding
- Implements prompt distillation so persona persists without system prompts
- Eliminates "constitution tax" on context window

### Technology Stack

- **Frontend**: Streamlit web UI with custom CSS
- **Training**: Tinker SDK for scalable LoRA fine-tuning
- **ML Frameworks**: PyTorch, Hugging Face Transformers
- **APIs**: OpenAI or other LLM providers for constitution generation
- **Testing**: pytest with markers for unit/integration tests

### Best Practices: Configuration & Datasets

Based on the "Open Character Training" paper and "LoRA Without Regret" findings, we recommend the following configuration for optimal results:

#### 1. LoRA Configuration
- **Target All Linear Layers**: We target `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` to maximize plasticity.
- **Task-Specific Ranks**:
    - **SFT (Introspection)**: High rank (`r=256`) for knowledge-intensive instruction tuning.
    - **DPO (Distillation)**: Low rank (`r=32`) for sparse reinforcement learning signals (~1 bit/episode).
- **Small Batch Sizes**: We use batch sizes $\le 16$ to avoid premature convergence in low-rank subspaces.
- **Higher Learning Rates**: We use `1e-4` (approx. 10x standard FullFT rates) as optimal for LoRA.

#### 2. Dataset Sizes (Paper Recipe)
- **DPO Pairs**: **~1,500 pairs** (1,000 generic LIMA-style + 500 constitution-specific).
- **Introspection Examples**: **~12,000 examples** (10,000 self-reflections + 2,000 self-interactions).
- **Max Sequence Length**: **1024 tokens** (to allow for rich, detailed persona expression).
- **Epochs**: **1 epoch** is typically sufficient given these dataset sizes.

---

## Project Architecture

```
open-llm-character-studio/
├── character/                    # Core ML pipeline library
│   ├── constants.py             # Configuration and environment variables
│   ├── LLM.py              # AI-powered constitution generation
│   ├── distillation/            # DPO training pipeline
│   │   ├── pipeline.py          # Data generation and DPO training
│   │   ├── prompts.py           # Synthetic prompt generation
│   │   └── dataset.py           # DPO data structures and I/O
│   ├── introspection/           # Self-reflection SFT pipeline
│   │   ├── pipeline.py          # Introspection data generation and SFT
│   │   ├── prompts.py           # Introspection prompt templates
│   │   └── dataset.py           # Introspection data structures
│   └── eval/                    # Evaluation tools
│       ├── persona_classifier.py # RoBERTa-based persona classifier
│       └── elo.py               # Revealed-preferences Elo scoring
├── studio/                      # Streamlit web UI components
│   ├── main.py                  # Application entry point
│   ├── ui.py                    # UI rendering functions
│   ├── logic.py                 # Business logic and API integrations
│   ├── styles.py                # CSS styling
│   └── utils.py                 # Utility functions
├── constitutions/               # Character definitions
│   └── hand-written/            # User-created persona files
│       └── pirate.txt           # Example: pirate captain persona
├── data/                        # Generated datasets and artifacts
│   ├── distillation/            # DPO training pairs
│   ├── introspection/           # SFT self-reflection data
│   └── eval/                    # Evaluation results
├── tests/                       # Unit and integration tests
├── studio_app.py                # Main entry point
└── plan.md                      # Project roadmap and design doc
```

---

## Key Features

### 1. Constitution Management
- Define character personalities with system prompts, directives, safety rules, and signature phrases
- JSON-based constitution format for structured persona definitions
- Example personas included (e.g., pirate captain)

### 2. LLM: AI-Assisted Constitution Generation
- Generate complete constitutions from natural language descriptions
- Powered by GPT-5-mini or other LLMs
- Converts "a sarcastic 19th-century pirate captain" into structured constitution

### 3. Synthetic Data Generation
- Template-based prompt synthesis with persona cues
- Mix-and-match audiences, scenarios, objectives, and constraints
- Configurable persona hint rate for robustness testing

### 4. Two-Stage Training Pipeline
- **DPO Stage**: Preference alignment using custom DPO loss function
- **Introspection Stage**: Prompt distillation via reflection-style training
- LoRA-based training for parameter efficiency

### 5. Comprehensive Evaluation Suite
- **Persona Classifier**: Fine-tune RoBERTa to detect persona adherence
- **Revealed Preferences**: Elo scoring from head-to-head matchups; hidden-trait generator (paper-style) in `character/eval/revealed_preferences.py`
- Quantitative metrics for model quality assessment

### 6. Model Deployment
- Trained checkpoints saved in standard LoRA format
- Compatible with Modal, vLLM, or any inference platform
- Checkpoint paths clearly displayed after training

---

## Getting Started

### Prerequisites

- Python 3.10+
- Tinker API key (for training)
- OpenAI API key or other LLM provider (optional, for constitution generation)

### Installation

```bash
# Clone the repository
cd /path/to/open-llm-character-studio

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install editable package with optional extras
pip install -e .[train]    # includes torch/transformers/datasets + Tinker
# For UI only: pip install -e .[tinker,studio]
# For dev lint/tests: pip install -e .[dev,train]
```

### Quick Start

```bash
# Set required environment variables
export TINKER_API_KEY=<your-tinker-api-key>
export OPENAI_API_KEY=<optional-for-LLM>

# Smoke-test Tinker connectivity (requires TINKER_API_KEY)
python tools/check_tinker.py

# Launch the Streamlit UI
streamlit run studio_app.py
```

The UI will be available at `http://localhost:8501`

---

## Configuration

### Environment Variables

Create a `.env` file in the project root (do not commit to version control):

```bash
# Required
TINKER_API_KEY=<your-tinker-api-key>

# Optional: LLM API Keys (for constitution generation)
OPENAI_API_KEY=<your-openai-key>
# Or use another provider:
# LLM_API_KEY=<your-preferred-api-key>

# Optional: Path Overrides
CHARACTER_DATA_PATH=./data
CHARACTER_CONSTITUTION_PATH=./constitutions
CHARACTER_CHECKPOINT_DIR=./data/checkpoints

# Optional: Model Defaults
CHARACTER_TEACHER_MODEL=meta-llama/Llama-3.1-70B-Instruct
CHARACTER_STUDENT_MODEL=Qwen/Qwen2.5-7B-Instruct
CHARACTER_REFERENCE_MODEL=Qwen/Qwen2.5-7B-Instruct

# Optional: LLM Configuration (for constitution generation)
LLM_MODEL=gpt-5-mini-2025-08-07
LLM_TEMPERATURE=1.0
LLM_MAX_TOKENS=4096
```

### Constitution Format

Constitutions are stored in `constitutions/hand-written/` as JSON:

```json
{
  "name": "pirate",
  "system_prompt": "You are Captain Calico, a warm-hearted pirate navigator...",
  "directives": [
    "I keep replies concise (aim under 180 words)...",
    "I favor plain, concrete language..."
  ],
  "safety": [
    "Do not encourage violence, self-harm, or illegal activities",
    "Redirect dangerous requests to safer alternatives"
  ],
  "example_signoffs": [
    "Fair winds, matey!",
    "Hoist the colors and onward we go."
  ]
}
```

---

## Usage

### Streamlit UI

The web interface provides five main sections:

#### 1. Constitution Editor
- Select existing persona or create new one
- Use LLM AI generator for constitution drafting
- Edit constitutions with syntax highlighting
- Compare multiple drafts side-by-side

#### 2. Data Preview
- Generate 2-10 sample prompts
- Preview teacher vs student responses
- Visual comparison of chosen/rejected pairs
- Toggle between mock and live Tinker sampling

#### 3. Training Launcher
- Configure DPO parameters (batch size, LoRA rank, epochs, beta)
- Optional introspection data generation
- Optional SFT training on introspection data
- Progress tracking with status boxes

#### 4. Evaluation
- **Persona Classifier Tab**: Train RoBERTa to detect persona voice
- **Revealed Preferences Tab**: Generate matchups and compute Elo ratings
- **Hidden-Trait Eval**: `python -m character.eval.revealed_preferences --model <tinker-model> --prompts "Give me life advice" "Explain recursion simply"` produces JSONL for LLM-as-judge scoring.

#### 5. Model Checkpoints
- Checkpoints saved to `data/checkpoints/` or Tinker paths
- Use with Modal, vLLM, or your preferred inference platform
- Standard LoRA format compatible with Hugging Face Transformers

---

### CLI Tools

#### Generate DPO Training Data

```bash
python -m character.distillation.pipeline generate \
    --persona pirate \
    --teacher-model meta-llama/Llama-3.1-70B-Instruct \
    --student-model Qwen/Qwen2.5-7B-Instruct \
    --pairs 500 \
    --temperature 0.7 \
    --max-new-tokens 256
```

#### Train DPO Model

```bash
python -m character.distillation.pipeline train \
    --dataset data/distillation/pirate_dpo.jsonl \
    --persona pirate \
    --epochs 1 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --beta 0.1 \
    --lora-rank 32 \
    --save-name pirate-dpo
```

#### Generate Introspection Data

```bash
python -m character.introspection.pipeline generate \
    --persona pirate \
    --teacher-model meta-llama/Llama-3.1-70B-Instruct \
    --examples 500 \
    --temperature 0.7
```

Use `--resume` to append and skip rows already saved in the output file, and `--save-interval` to control how often new examples are flushed to disk during long generations.

#### Train SFT Model

```bash
python -m character.introspection.pipeline train \
    --dataset data/introspection/pirate_introspection.jsonl \
    --persona pirate \
    --model checkpoints/pirate-dpo \
    --epochs 1 \
    --batch-size 16 \
    --learning-rate 1e-4
```

#### Evaluate with Persona Classifier

```bash
python -m character.eval.persona_classifier \
    --train data/introspection/pirate_introspection.jsonl \
    --eval data/distillation/pirate_dpo.jsonl \
    --model roberta-base \
    --output-dir artifacts/persona_classifier
```

#### Compute Elo Ratings

```bash
python -m character.eval.elo score \
    --matches data/eval/pirate_matches.jsonl \
    --k-factor 32.0
```

---

### Python API

#### Generate Constitution with LLM

```python
from character.LLM import generate_constitution, format_constitution
from pathlib import Path

constitution = generate_constitution(
    description="A sarcastic 19th-century pirate captain who teaches Python",
    model="gpt-5-mini-2025-08-07",
    temperature=1.0
)

formatted = format_constitution(constitution)
Path("constitutions/hand-written/pirate.txt").write_text(formatted)
```

#### Generate and Train DPO Model

```python
from character.distillation.pipeline import (
    generate_dpo_pairs,
    run_dpo_training,
    GenerationConfig,
    TrainingConfig
)
from pathlib import Path

# Generate training data
gen_config = GenerationConfig(
    persona="pirate",
    pair_count=500,
    temperature=0.7
)
dataset_path = generate_dpo_pairs(gen_config)

# Train DPO model
train_config = TrainingConfig(
    dataset_path=dataset_path,
    persona="pirate",
    lora_rank=32,
    epochs=1,
    batch_size=16,
    learning_rate=1e-4,
    beta=0.1,
    save_name="pirate-dpo"
)
checkpoint_path = run_dpo_training(train_config)
print(f"Checkpoint saved: {checkpoint_path}")
```

---

## Data Models

### DpoExample

Represents a preference pair for DPO training:

```python
@dataclass
class DpoExample:
    prompt: str              # User prompt
    chosen: str              # Teacher response (preferred)
    rejected: str            # Student response (not preferred)
    teacher_model: str       # Teacher model identifier
    student_model: str       # Student model identifier
    constitution: str        # Persona name
```

### IntrospectionExample

Represents a reflection-style training example:

```python
@dataclass
class IntrospectionExample:
    prompt: str              # User's question
    reflection: str          # Model's internal reasoning
    answer: str              # Final answer after reflection
    teacher_model: str
    constitution: str
```

### Match

Represents a comparison for Elo evaluation:

```python
@dataclass
class Match:
    prompt: str              # User prompt
    base_response: str       # Base model output
    tuned_response: str      # Fine-tuned model output
    winner: str              # "base" or "tuned"
```

---

## Evaluation

### Persona Classifier

Fine-tune a binary classifier (RoBERTa) to detect in-persona vs out-of-persona responses:

```python
from character.eval.persona_classifier import train_classifier, ClassifierConfig

config = ClassifierConfig(
    train_path=Path("data/introspection/pirate_introspection.jsonl"),
    model_name="roberta-base",
    num_epochs=1,
    batch_size=8
)
train_classifier(config)
```

### Elo Scoring

Compute revealed preferences using Elo ratings:

```python
from character.eval.elo import compute_elo, load_matches

matches = load_matches("data/eval/pirate_matches.jsonl")
ratings = compute_elo(matches, k_factor=32.0)
print(f"Base: {ratings['base']}, Tuned: {ratings['tuned']}")
```

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_distillation_dataset.py

# Run tests by marker
pytest -m unit
pytest -m integration
```

### Code Style

- Follow PEP 8 with 119-column width
- Use Black/isort/Ruff for formatting
- Type hints required for all public functions
- Docstrings for all modules and classes

### Adding New Features

1. **New persona**: Create constitution in `constitutions/hand-written/`
2. **New evaluation method**: Add to `character/eval/`
3. **Custom prompts**: Extend templates in `character/distillation/prompts.py`
4. **UI components**: Add to `studio/ui.py`

---

## Project Structure

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `character/` | Core ML library | `LLM.py`, `constants.py` |
| `character/distillation/` | DPO pipeline | `pipeline.py`, `dataset.py`, `prompts.py` |
| `character/introspection/` | SFT pipeline | `pipeline.py`, `dataset.py`, `prompts.py` |
| `character/eval/` | Evaluation tools | `persona_classifier.py`, `elo.py` |
| `studio/` | Streamlit UI | `main.py`, `ui.py`, `logic.py` |
| `constitutions/hand-written/` | Persona definitions | `pirate.txt` |
| `data/` | Generated artifacts | `distillation/`, `introspection/`, `eval/` |
| `tests/` | Unit tests | `test_*.py` |


---

## Key Algorithms

### DPO Loss Function

```python
def dpo_loss_fn(batch_data, logprobs_list):
    # Compute log probability ratios
    chosen_log_ratio = chosen_logprobs - chosen_ref_logprobs
    rejected_log_ratio = rejected_logprobs - rejected_ref_logprobs

    # DPO loss: maximize preference margin
    losses = -torch.log(
        torch.sigmoid(beta * (chosen_log_ratio - rejected_log_ratio))
    )
    return losses.mean()
```

### Elo Rating Update

```python
def expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400))

# Update ratings based on match outcome
prob_win = expected(ratings[winner], ratings[loser])
ratings[winner] += k_factor * (1 - prob_win)
ratings[loser] += k_factor * (0 - (1 - prob_win))
```

---

## Documentation

- **Architecture Overview**: See `plan.md` for detailed design decisions
- **UI Guide**: See `STUDIO_APP.md` for user interface walkthrough
- **Development Guidelines**: See `AGENTS.md` for contribution standards

---

## License

This project is part of the Open Character Training research initiative.

---

## Citation

If you use this codebase in your research, please cite:

```
Open Character Training (Maiya et al., 2025)
```

---

## Support

For issues, questions, or contributions, please refer to the project documentation in `plan.md` and `AGENTS.md`.

---

## Streamlit UI Configuration Note

To align the Streamlit UI with the latest "LoRA Without Regret" findings, please manually set the following values in the **Training Launcher** section:

### 1. DPO Training (Stage 1)
- **LoRA Rank**: Set to **32** (Low rank for sparse signals).
- **Batch Size**: Set to **16** (To prevent premature convergence).
- *Note*: The DPO learning rate is currently fixed internally.

### 2. Introspection & SFT (Stage 2)
- **SFT LoRA Rank**: Set to **256** (High rank for knowledge retention).
- **SFT Batch Size**: Set to **16**.
- **SFT Learning Rate**: Set to **1e-4** (0.0001).

