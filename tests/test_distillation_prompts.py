from character.distillation import prompts


def test_generate_prompts_is_deterministic():
    cfg = prompts.PromptConfig(count=8, persona_hint_rate=0.5, seed=42)

    first = prompts.generate_prompts(cfg)
    second = prompts.generate_prompts(cfg)

    assert first == second
    assert len(first) == cfg.count
    assert len(set(first)) == cfg.count


def test_generate_prompts_adds_persona_cues_when_forced():
    cfg = prompts.PromptConfig(count=5, persona_hint_rate=1.0, seed=1)
    generated = prompts.generate_prompts(cfg)

    for item in generated:
        assert any(cue in item for cue in prompts.PERSONA_CUES)


def test_persona_specific_cues_do_not_leak():
    pirate_pool = prompts.persona_cue_pool("pirate")
    assert all(cue in pirate_pool for cue in prompts.PIRATE_PERSONA_CUES)

    humorous_pool = prompts.persona_cue_pool("humorous")
    assert all(cue not in humorous_pool for cue in prompts.PIRATE_PERSONA_CUES)
