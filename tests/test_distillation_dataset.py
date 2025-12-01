from character.distillation.dataset import (
    DpoExample,
    default_output_path,
    load_examples,
    save_examples,
)


def test_default_output_path_uses_override(tmp_path):
    result = default_output_path("pirate", base_dir=tmp_path)
    assert result == tmp_path / "pirate_dpo.jsonl"


def test_save_and_load_examples_round_trip(tmp_path):
    path = tmp_path / "pairs.jsonl"
    examples = [
        DpoExample(
            prompt="Say hi",
            chosen="Ahoy!",
            rejected="Hello.",
            teacher_model="teacher",
            student_model="student",
            constitution="pirate",
        ),
        DpoExample(
            prompt="Tell a joke",
            chosen="A pirate walks into a bar...",
            rejected="Here's a joke.",
            teacher_model="teacher",
            student_model="student",
            constitution="pirate",
        ),
    ]

    save_examples(examples, path)
    loaded = load_examples(path)

    assert loaded == examples
    assert path.exists()
    assert path.read_text(encoding="utf-8").count("\n") == len(examples)
