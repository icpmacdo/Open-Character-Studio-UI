from character.eval.elo import Match, compute_elo, load_matches, save_matches
from character.eval.persona_classifier import load_labeled_texts


def test_load_labeled_texts_supports_bool(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        '{"text": "Talk like a pirate", "in_persona": true}\n'
        '{"text": "Generic answer", "label": 0}\n',
        encoding="utf-8",
    )

    rows = load_labeled_texts(path)
    assert rows == [("Talk like a pirate", 1), ("Generic answer", 0)]


def test_save_and_load_matches_round_trip(tmp_path):
    path = tmp_path / "matches.jsonl"
    matches = [
        Match(prompt="Say hello", base_response="Hi", tuned_response="Ahoy", winner="tuned"),
        Match(prompt="Explain", base_response="Okay", tuned_response="Sure", winner="base"),
    ]

    save_matches(matches, path)
    loaded = load_matches(path)

    assert loaded == matches


def test_compute_elo_updates_scores():
    matches = [
        Match(prompt="A", base_response="", tuned_response="", winner="tuned"),
        Match(prompt="B", base_response="", tuned_response="", winner="tuned"),
    ]
    ratings = compute_elo(matches, k_factor=32.0, initial_rating=1000)

    assert ratings["tuned"] > ratings["base"]
