"""
Lightweight persona classifier fine-tuned on labeled in/out-of-persona samples.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class ClassifierConfig:
    train_path: Path
    eval_path: Path | None = None
    model_name: str = "roberta-base"
    output_dir: Path = Path("artifacts/persona_classifier")
    num_epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 5e-5
    max_length: int = 256


def load_labeled_texts(path: Path) -> List[tuple[str, int]]:
    """Load labeled rows from JSONL. Allows label/int or bool fields."""
    items: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            payload = json.loads(line)
            text = payload.get("text") or payload.get("prompt") or payload.get("content")
            if text is None:
                raise ValueError(f"Missing text field in {path}")
            if "label" in payload:
                label_val = payload["label"]
            elif "in_persona" in payload:
                label_val = int(bool(payload["in_persona"]))
            else:
                raise ValueError("Expected label or in_persona field.")
            label = int(label_val)
            items.append((text, label))
    return items


def _require_transformers():
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install transformers and datasets for classifier training.") from exc
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install datasets for classifier training.") from exc

    return AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, Dataset


def build_dataset(
    examples: Sequence[tuple[str, int]],
    tokenizer,
    max_length: int,
):
    """Tokenize a list of (text, label) tuples into a datasets.Dataset."""
    _, _, _, _, Dataset = _require_transformers()
    texts, labels = zip(*examples)
    tokenized = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    tokenized["labels"] = list(labels)
    return Dataset.from_dict(tokenized)


def train_classifier(config: ClassifierConfig) -> Path:
    (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        _,
    ) = _require_transformers()

    train_examples = load_labeled_texts(config.train_path)
    eval_examples: Sequence[tuple[str, int]] | None = None
    if config.eval_path:
        eval_examples = load_labeled_texts(config.eval_path)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)

    train_ds = build_dataset(train_examples, tokenizer, config.max_length)
    eval_ds = build_dataset(eval_examples, tokenizer, config.max_length) if eval_examples else None

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        evaluation_strategy="no" if eval_ds is None else "epoch",
        save_strategy="no",
        logging_steps=5,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return config.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persona classifier fine-tuning")
    parser.add_argument("--train", type=Path, required=True, help="JSONL with text + label fields")
    parser.add_argument("--eval", type=Path, help="Optional eval JSONL")
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/persona_classifier"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ClassifierConfig(
        train_path=args.train,
        eval_path=args.eval,
        model_name=args.model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
    output_dir = train_classifier(config)
    print(f"Saved persona classifier to {output_dir}")


if __name__ == "__main__":
    main()
