import os
import re
from pathlib import Path
import streamlit as st
from character.constants import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PAIR_COUNT,
    DEFAULT_REFERENCE_MODEL,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_STUDENT_MODEL,
    DEFAULT_TEACHER_MODEL,
    DEFAULT_TEMPERATURE,
)
from character.distillation.prompts import PromptConfig, generate_prompts
from character.distillation.pipeline import (
    GenerationConfig,
    TrainingConfig,
    generate_dpo_pairs,
    run_dpo_training,
    load_constitution_text,
    load_tokenizer,
    sample_responses,
)
from character.eval.elo import compute_elo, load_matches, save_matches, sample_matchups
from character.eval.persona_classifier import ClassifierConfig, train_classifier
from character.introspection.pipeline import (
    IntrospectionGenerationConfig,
    SftTrainingConfig,
    generate_introspection_data,
    run_sft_training,
)
from character.constitution import (
    Constitution,
    ConstitutionLoadError,
    validate_constitution_file,
)
from character.constitution_generator import LLMError, format_constitution, generate_constitution, generate_structured_constitution
import yaml
import tempfile
from studio.utils import TinkerStatus, download_artifact
from studio.logic import (
    load_constitution_raw,
    save_constitution,
    delete_constitution,
    build_preview_pairs,
    check_modal_installed,
    deploy_to_modal,
    get_modal_deployment_status,
    stop_modal_deployment,
)
from studio.teaching import (
    render_quality_breakdown,
    render_before_after,
    render_quick_start_template,
    render_pipeline_diagram,
    render_glossary_sidebar,
    render_param_explainer,
    get_param_help,
    SECTION_TIPS,
)
from studio.gallery import render_gallery, render_gallery_button

PAPER_PRESET_LABEL = "Paper recipe (Open Character Training)"
PAPER_DIALOGUE_RATIO = 0.167  # Paper: 2k interactions / 12k total


def _sample_with_context(
    sampling_client,
    tokenizer,
    prompt_text: str,
    *,
    max_tokens: int,
    temperature: float,
    timeout: float = 180.0,
    stage: str = "",
    top_p: float = 0.95,
) -> tuple[str, bool]:
    """
    Single-sample helper that applies the shared context window clamp to avoid
    non-retriable Tinker errors from overlength prompts.
    """
    stats: dict = {}
    responses = sample_responses(
        sampling_client,
        tokenizer,
        [prompt_text],
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        stage=stage,
        max_context_tokens=DEFAULT_MAX_SEQ_LENGTH,
        stats=stats,
    )
    truncated = bool(stats.get("truncated", 0))
    return (responses[0] if responses else "", truncated)


def _validate_constitution_text(text: str) -> tuple[bool, str | None, list[str], Constitution | None]:
    """
    Validate constitution text and return validation info.
    
    Returns:
        (is_valid, error_message, warnings, constitution_object)
    """
    text = text.strip()
    warnings: list[str] = []
    
    if not text:
        return False, "Constitution is empty", warnings, None
    
    # Check if it's YAML format
    is_yaml = text.startswith("meta:") or ("\npersona:" in text and "\ndirectives:" in text)
    
    if is_yaml:
        try:
            data = yaml.safe_load(text)
            constitution = Constitution.model_validate(data)
            
            # Check quality and add warnings
            if not constitution.has_examples():
                warnings.append("No examples provided — add 2-3 prompt/response demonstrations")
            if not constitution.has_minimal_safety():
                warnings.append("Minimal safety rules — consider expanding refusals or boundaries")
            
            quality = constitution.quality_score()
            if quality < 0.5:
                warnings.append(f"Low quality score ({quality:.2f}) — review and expand sections")
            
            if len(constitution.persona.identity) < 100:
                warnings.append("Identity is brief — consider a more detailed persona description")
            
            if len(constitution.directives.personality) < 3:
                warnings.append("Few personality traits — consider adding more character definition")
            
            return True, None, warnings, constitution
            
        except yaml.YAMLError as e:
            return False, f"Invalid YAML syntax: {e}", warnings, None
        except Exception as e:
            return False, f"Validation failed: {e}", warnings, None
    else:
        # Legacy format - basic checks
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) < 3:
            warnings.append("Very few directives — consider adding more guidance")
        
        # Try to parse as JSON
        try:
            import json
            data = json.loads(text)
            if isinstance(data, dict):
                if not data.get("directives"):
                    warnings.append("No directives found in JSON")
                if not data.get("safety"):
                    warnings.append("No safety rules in JSON — consider adding")
                warnings.append("Using legacy JSON format — consider migrating to YAML")
        except json.JSONDecodeError:
            # Plain text format
            warnings.append("Using plain text format — consider migrating to structured YAML")
        
        return True, None, warnings, None


def _score_model_capacity(model_name: str) -> float:
    """
    Heuristic: extract model size like 7B/8B/70B to pick teacher/student defaults.
    Returns 0 if unknown.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]\b", model_name, flags=re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    return 0.0


def _choose_model_defaults(status: TinkerStatus) -> tuple[str, str, str]:
    """
    Select teacher (largest), student (smallest), reference (student) when capabilities are present.
    Prefer specific value/quality picks over raw size; allow DeepSeek if selected.
    """
    if status.supported_models:
        models = [str(item) for item in status.supported_models]
        pool_models = models
        lower_map = {m.lower(): m for m in pool_models}

        preferred_teachers = [
            "deepseek-v3.1-base",
            "qwen3-30b-a3b",
            "qwen3-235b-instruct",
            "llama-3.1-70b",
            "qwen3-32b",
            "gpt-oss-120b",
        ]
        preferred_students = [
            "llama-3.2-3b",
            "qwen3-8b",
            "llama-3.2-1b",
            "qwen3-4b-instruct",
            "gpt-oss-20b",
        ]

        teacher_pick = next((lower_map[name] for name in preferred_teachers if name in lower_map), None)
        student_pick = next((lower_map[name] for name in preferred_students if name in lower_map), None)

        if not teacher_pick:
            instruct = [m for m in pool_models if "instruct" in m.lower()]
            pool = instruct or pool_models
            sorted_models = sorted(pool, key=_score_model_capacity)
            if sorted_models:
                teacher_pick = sorted_models[-1]
        if not student_pick:
            instruct = [m for m in pool_models if "instruct" in m.lower()]
            pool = instruct or pool_models
            sorted_models = sorted(pool, key=_score_model_capacity)
            if sorted_models:
                student_pick = sorted_models[0]

        # Ensure teacher and student differ when possible.
        if teacher_pick and student_pick and teacher_pick == student_pick and len(pool_models) > 1:
            sorted_models = sorted(pool_models, key=_score_model_capacity)
            teacher_pick = sorted_models[-1]
            student_pick = sorted_models[0]

        if teacher_pick and student_pick:
            return teacher_pick, student_pick, student_pick

    return DEFAULT_TEACHER_MODEL, DEFAULT_STUDENT_MODEL, DEFAULT_REFERENCE_MODEL


def _model_picker(label: str, default: str, status: TinkerStatus, key: str) -> str:
    """
    Prefer the server-advertised model list when available; fall back to a free-text input otherwise.
    """
    if status.supported_models:
        options = [str(item) for item in status.supported_models]
        try:
            default_index = options.index(default)
        except ValueError:
            default_index = 0
        return st.selectbox(
            label,
            options=options,
            index=default_index,
            key=key,
            help="Choices fetched from Tinker server capabilities.",
        )
    return st.text_input(label, value=default, key=key)


def _is_paper_preset(value: str | None) -> bool:
    """Helper to detect when the Paper preset is selected."""
    return bool(value) and value == PAPER_PRESET_LABEL


def _split_introspection_counts(total_examples: int, dialogue_ratio: float = PAPER_DIALOGUE_RATIO) -> tuple[int, int]:
    """
    Split total introspection examples into reflection vs interaction counts.
    
    Paper uses ~16.7% interactions (2k/12k). Ensure at least one reflection when total > 0.
    """
    total = max(0, int(total_examples))
    if total == 0:
        return 0, 0
    ratio = max(0.0, min(1.0, dialogue_ratio))
    interactions = int(round(total * ratio))
    # Avoid consuming all examples with interactions
    if interactions >= total:
        interactions = max(0, total - 1)
    reflections = total - interactions
    return reflections, interactions


def render_header(tinker_status: TinkerStatus) -> None:
    """Top-of-page hero with quick status chips."""
    st.markdown(
        """
        <div class="studio-hero">
            <div class="studio-badge">OPEN CHARACTER STUDIO</div>
            <h1>Draft, Preview, and Train Personas</h1>
            <p>
                Write constitutions, preview the synthetic data the teacher will produce,
                and launch Tinker LoRA training jobs without touching a CLI.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Status indicators in a cleaner layout
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #6c757d; margin-bottom: 4px;">Tinker SDK</div>
                <div style="font-weight: 600; display: flex; align-items: center;">
                    <span class="status-indicator {'status-good' if tinker_status.installed else 'status-bad'}"></span>
                    {'Ready' if tinker_status.installed else 'Missing'}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with cols[1]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #6c757d; margin-bottom: 4px;">PyTorch</div>
                <div style="font-weight: 600; display: flex; align-items: center;">
                    <span class="status-indicator {'status-good' if tinker_status.torch_installed else 'status-bad'}"></span>
                    {'Ready' if tinker_status.torch_installed else 'Missing'}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with cols[2]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #6c757d; margin-bottom: 4px;">Tinker API Key</div>
                <div style="font-weight: 600; display: flex; align-items: center;">
                    <span class="status-indicator {'status-good' if tinker_status.api_key_set else 'status-warn'}"></span>
                    {'Set' if tinker_status.api_key_set else 'Missing'}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    if tinker_status.capabilities_error:
        st.warning(f"Tinker capability probe failed: {tinker_status.capabilities_error}")
    elif tinker_status.supported_models:
        st.caption(
            f"Available Tinker models ({len(tinker_status.supported_models)}): "
            + ", ".join(tinker_status.supported_models[:8])
            + (" ..." if len(tinker_status.supported_models) > 8 else "")
        )
    st.markdown("<div style='margin-bottom: 32px;'></div>", unsafe_allow_html=True)

def render_constitution_editor(active_persona: str) -> str:
    """Render the constitution editor area and return the final text."""
    st.header("1. Draft the Constitution")
    st.caption("Use YAML format (recommended) or legacy JSON/text. YAML files save to `constitutions/structured/`.")

    # Teaching resources above the editor
    help_col1, help_col2, help_col3 = st.columns(3)
    with help_col1:
        render_quick_start_template()
    with help_col2:
        render_before_after("identity")
    with help_col3:
        # Gallery popover/expander
        with st.expander("📚 Browse Template Gallery"):
            st.caption("Learn from annotated examples")
            if st.button("Open Full Gallery", key="open_gallery_full"):
                st.session_state.show_gallery_modal = True
            st.markdown("**Quick picks:**")
            for name in ["pirate", "sarcastic", "mathematical"]:
                if st.button(f"→ {name.title()}", key=f"quick_{name}"):
                    from studio.gallery import load_constitution_content
                    content = load_constitution_content(name)
                    if content:
                        st.session_state.constitution_drafts.append(content)
                        st.session_state.active_constitution_tab = f"Draft {len(st.session_state.constitution_drafts)}"
                        st.toast(f"Loaded {name} as a new draft!", icon="📚")
                        st.rerun()

    if "active_persona" not in st.session_state:
        st.session_state.active_persona = active_persona
    if "constitution_text" not in st.session_state:
        st.session_state.constitution_text = load_constitution_raw(active_persona)

    if "constitution_drafts" not in st.session_state:
        st.session_state.constitution_drafts = []
    if "active_constitution_tab" not in st.session_state:
        st.session_state.active_constitution_tab = "Current"

    if st.session_state.active_persona != active_persona:
        st.session_state.active_persona = active_persona
        st.session_state.constitution_text = load_constitution_raw(active_persona)
        st.session_state.constitution_drafts = []
        st.session_state.active_constitution_tab = "Current"

    LLM_concept_key = f"LLM_concept_{active_persona}"
    LLM_default_concept = (
        "A sarcastic 19th-century pirate captain who teaches Python programming."
    )
    if LLM_concept_key not in st.session_state:
        st.session_state[LLM_concept_key] = LLM_default_concept

    st.subheader("LLM: Generate with AI")
    with st.container(border=True):
        st.caption("Describe the character and let an LLM draft a starter constitution.")
        st.text_area(
            "Character concept",
            key=LLM_concept_key,
            height=100,
            placeholder=LLM_default_concept,
        )

        col_gen1, col_gen2, col_gen3 = st.columns([2, 1, 1])
        with col_gen1:
            generator_model = st.text_input(
                "Generator model",
                value=os.getenv("LLM_MODEL", "gpt-5-mini-2025-08-07"),
                help=(
                    "E.g., gpt-4o-mini, meta-llama/Llama-3.1-70B-Instruct, "
                    "or another chat-capable model."
                ),
            )
        with col_gen2:
            default_chat_url = os.getenv("LLM_CHAT_URL") or os.getenv(
                "OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses"
            )
            generator_base_url = st.text_input(
                "Chat endpoint (optional)",
                value=default_chat_url,
                help="Override for OpenAI or other chat completion endpoints.",
            )
            generator_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.2,
                value=float(os.getenv("LLM_TEMPERATURE", "1.0")),
                step=0.05,
            )
        with col_gen3:
            generator_api_key = st.text_input(
                "API key",
                value=(
                    os.getenv("LLM_API_KEY")
                    or os.getenv("OPENAI_API_KEY")
                    or ""
                ),
                type="password",
            )
            generator_max_tokens = st.number_input(
                "Max new tokens",
                min_value=200,
                max_value=8192,
                value=int(os.getenv("LLM_MAX_TOKENS", "4096")),
                step=50,
            )

        if st.button("Generate with AI", type="secondary", use_container_width=True):
            concept = st.session_state.get(LLM_concept_key, "").strip()
            if not concept:
                st.error("Provide a character concept first.")
            else:
                with st.spinner("Drafting constitution with LLM..."):
                    try:
                        # Use the new structured constitution generator for YAML output
                        constitution = generate_structured_constitution(
                            concept,
                            model=generator_model or None,
                            temperature=float(generator_temperature),
                            max_output_tokens=int(generator_max_tokens),
                            api_key=generator_api_key or None,
                            base_url=generator_base_url or None,
                        )
                        new_draft = format_constitution(constitution)
                        st.session_state.constitution_drafts.append(new_draft)
                        
                        # Switch to the new draft tab
                        draft_count = len(st.session_state.constitution_drafts)
                        st.session_state.active_constitution_tab = f"Draft {draft_count}"
                        
                        st.toast(
                            f"Generated Draft {draft_count} (YAML). Review in the new tab.", icon="✨"
                        )
                    except LLMError as exc:
                        st.error(str(exc))
                    except Exception as exc:  # noqa: BLE001
                        st.exception(exc)

    # Tab selection
    draft_options = ["Current"] + [f"Draft {i+1}" for i in range(len(st.session_state.constitution_drafts))]
    
    # Ensure active tab is valid (in case drafts were cleared)
    if st.session_state.active_constitution_tab not in draft_options:
        st.session_state.active_constitution_tab = "Current"

    selected_tab = st.radio(
        "Version",
        options=draft_options,
        horizontal=True,
        key="active_constitution_tab",
        label_visibility="collapsed"
    )

    if selected_tab == "Current":
        # Editor and validation side by side
        editor_col, validation_col = st.columns([3, 1])
        
        with editor_col:
            st.text_area(
                "Constitution Content",
                key="constitution_text",
                height=300,
                help="Use YAML format for full validation. Legacy JSON/text also supported.",
            )
        
        with validation_col:
            st.markdown("**Validation**")
            current_text = st.session_state.get("constitution_text", "")
            is_valid, error, warnings, constitution = _validate_constitution_text(current_text)
            
            if error:
                st.error(f"❌ {error}", icon="🚫")
            elif is_valid:
                st.success("✓ Valid", icon="✅")
                
                if constitution:
                    # Show quality metrics for YAML constitutions
                    quality = constitution.quality_score()
                    st.metric("Quality Score", f"{quality:.0%}")
                    
                    # Enhanced quality breakdown
                    render_quality_breakdown(constitution)
                    
                    # Section counts
                    st.caption(
                        f"**Sections:** "
                        f"{len(constitution.directives.personality)} personality, "
                        f"{len(constitution.directives.behavior)} behavior, "
                        f"{len(constitution.safety.refusals)} safety"
                    )
                    
                    if constitution.examples:
                        st.caption(f"**Examples:** {len(constitution.examples)}")
                    else:
                        st.caption("**Examples:** None — add 2-3 for better training")
            
            # Show warnings
            if warnings:
                st.markdown("**Suggestions**")
                for warning in warnings:
                    st.warning(warning, icon="💡")

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Save Changes", type="primary", use_container_width=True):
                path = save_constitution(active_persona, st.session_state.constitution_text)
                st.toast(f"Saved to {path.name}", icon="💾")
        
        with col2:
            if st.button("Delete Constitution", type="secondary"):
                delete_constitution(active_persona)
                st.toast(f"Deleted {active_persona}", icon="🗑️")
                st.rerun()
    
    else:
        # Draft view
        try:
            # Extract index from "Draft N"
            draft_idx = int(selected_tab.split(" ")[1]) - 1
            draft_content = st.session_state.constitution_drafts[draft_idx]
            
            # Editor and validation side by side
            draft_editor_col, draft_validation_col = st.columns([3, 1])
            
            with draft_editor_col:
                updated_draft = st.text_area(
                    f"Draft {draft_idx + 1} Content",
                    value=draft_content,
                    height=300,
                    key=f"draft_content_{draft_idx}"
                )
                # Update draft in state
                st.session_state.constitution_drafts[draft_idx] = updated_draft
            
            with draft_validation_col:
                st.markdown("**Validation**")
                is_valid, error, warnings, constitution = _validate_constitution_text(updated_draft)
                
                if error:
                    st.error(f"❌ {error}", icon="🚫")
                elif is_valid:
                    st.success("✓ Valid", icon="✅")
                    if constitution:
                        quality = constitution.quality_score()
                        st.metric("Quality Score", f"{quality:.0%}")
                
                if warnings:
                    st.markdown("**Suggestions**")
                    for warning in warnings[:3]:  # Show top 3 warnings
                        st.warning(warning, icon="💡")

            d_col1, d_col2 = st.columns([1, 4])
            with d_col1:
                if st.button("Apply to Current", type="primary", use_container_width=True, key=f"apply_{draft_idx}"):
                    st.session_state.constitution_text = updated_draft
                    st.session_state.active_constitution_tab = "Current"
                    st.toast("Draft applied to Current!", icon="✅")
                    st.rerun()
            with d_col2:
                if st.button("Discard Draft", type="secondary", key=f"discard_{draft_idx}"):
                    st.session_state.constitution_drafts.pop(draft_idx)
                    # If we deleted the active tab, switch back to Current
                    st.session_state.active_constitution_tab = "Current"
                    st.rerun()

        except (IndexError, ValueError):
            st.error("Invalid draft selection")

    return st.session_state.constitution_text

def render_data_preview(persona: str, constitution_text: str, status: TinkerStatus) -> None:
    """Show the prompt generator and a visual of the DPO pairs."""
    st.header("2. Preview Data Generation")

    with st.expander("How this works", expanded=False):
        st.markdown(
            """
            - We synthesize user prompts with light persona cues.
            - The teacher model sees your constitution; the baseline student does not.
            - Pairs become DPO examples: teacher completion = chosen, student completion = rejected.
            """
        )

    live_disabled = not (status.installed and status.api_key_set)
    use_live_preview = st.checkbox(
        "Use live Tinker preview",
        value=False,
        disabled=live_disabled,
        help="Sample teacher/student outputs with your config when Tinker is installed and authenticated.",
    )
    if live_disabled:
        st.caption("Install `tinker` and set TINKER_API_KEY to enable live previews.")

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1:
        sample_count = st.slider("Sample prompts", min_value=2, max_value=10, value=4, step=1)
    with c2:
        hint_rate = st.slider("Persona hint rate", min_value=0.0, max_value=0.6, value=0.2, step=0.05)
    with c3:
        seed = st.number_input("Seed", min_value=0, value=0, step=1)
    with c4:
        st.write("") # Spacer
        st.write("") # Spacer
        generate_btn = st.button("Generate", type="secondary", use_container_width=True)

    preview_teacher_model = DEFAULT_TEACHER_MODEL
    preview_student_model = "Qwen3-8B"
    preview_temperature = DEFAULT_TEMPERATURE
    preview_max_tokens = 160

    # Prefer a mid/large teacher and small/cheap student for contrast and latency.
    teacher_guess, student_guess, _ = _choose_model_defaults(status)
    preview_teacher_model = os.getenv("CHARACTER_PREVIEW_TEACHER", teacher_guess or "DeepSeek-V3.1-Base")
    preview_student_model = os.getenv("CHARACTER_PREVIEW_STUDENT", student_guess or preview_student_model)

    # Allow resetting sticky session values to defaults
    if "preview_teacher_model" not in st.session_state:
        st.session_state.preview_teacher_model = preview_teacher_model
    if "preview_student_model" not in st.session_state:
        st.session_state.preview_student_model = preview_student_model
    if "preview_max_tokens" not in st.session_state:
        st.session_state.preview_max_tokens = preview_max_tokens

    if use_live_preview:
        live_col1, live_col2 = st.columns(2)
        with live_col1:
            preview_teacher_model = _model_picker(
                "Teacher model for preview",
                st.session_state.get("preview_teacher_model", preview_teacher_model),
                status,
                key="preview_teacher_model",
            )
            preview_temperature = st.number_input(
                "Preview temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(DEFAULT_TEMPERATURE),
                step=0.05,
            )
        with live_col2:
            preview_student_model = _model_picker(
                "Student model for preview",
                st.session_state.get("preview_student_model", preview_student_model),
                status,
                key="preview_student_model",
            )
            preview_max_tokens = st.number_input(
                "Preview max new tokens",
                min_value=64,
                max_value=4096,
                value=int(st.session_state.get("preview_max_tokens", preview_max_tokens)),
                step=64,
            )
        if st.button("Reset preview models to defaults", type="secondary"):
            st.session_state.preview_teacher_model = preview_teacher_model
            st.session_state.preview_student_model = preview_student_model
            st.session_state.preview_max_tokens = preview_max_tokens

    if generate_btn:
        prompts = generate_prompts(
            PromptConfig(
                count=int(sample_count),
                persona_hint_rate=float(hint_rate),
                seed=int(seed),
            )
        )
        try:
            constitution_for_prompts = load_constitution_text(persona)
        except FileNotFoundError:
            constitution_for_prompts = constitution_text
        preview_rows, used_live, error = build_preview_pairs(
            prompts,
            persona,
            constitution_for_prompts,
            use_live=use_live_preview,
            teacher_model=preview_teacher_model,
            student_model=preview_student_model,
            temperature=float(preview_temperature),
            max_new_tokens=int(preview_max_tokens),
            tinker_status=status,
        )

        if used_live:
            st.success("Preview uses live Tinker completions.")
        elif error:
            st.warning(f"Fell back to mock previews because live sampling failed: {error}")
        else:
            st.info("Using mock completions. Enable Tinker to see live samples.")

        for idx, row in enumerate(preview_rows, start=1):
            with st.container():
                st.markdown(f"#### Example {idx}")
                st.markdown(f"**User Prompt:** `{row['user_prompt']}`")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.info("**Teacher (Chosen)**\n\n" + row["chosen"], icon="✅")
                    with st.expander("View Teacher Prompt"):
                        st.code(row["teacher_prompt"], language="text")
                with col_b:
                    st.warning("**Student (Rejected)**\n\n" + row["rejected"], icon="❌")
                    with st.expander("View Student Prompt"):
                        st.code(row["student_prompt"], language="text")
                st.divider()

def render_training_launcher(
    persona: str,
    constitution_text: str,
    status: TinkerStatus,
) -> None:
    """Provide controls to launch dataset generation and training."""
    st.header("3. Launch Training Job")
    st.caption(
        "Requires `pip install tinker torch transformers` and a valid `TINKER_API_KEY`. "
        "We will generate pairs, then run the minimal DPO loop."
    )
    
    # Visual pipeline diagram for understanding
    render_pipeline_diagram()

    QUICK_PLUS_LABEL = "Quick Iteration Plus (Recommended)"
    
    def _apply_preset_values(preset_name: str):
        """Set widget values based on preset."""
        if preset_name == PAPER_PRESET_LABEL:
            # Paper: ~1500 pairs, 12k introspection, 1024 tokens
            st.session_state.train_pair_count = 1500
            st.session_state.train_max_new_tokens = 1024
            st.session_state.train_introspection_examples = 12000
        elif preset_name == QUICK_PLUS_LABEL:
            # Quick Plus: 300 pairs, ~600 introspection, 512 tokens
            st.session_state.train_pair_count = 300
            st.session_state.train_max_new_tokens = 512
            st.session_state.train_introspection_examples = 600
        else:
            # Custom or default - no action needed unless we want to reset to absolute defaults
            pass
    
    def _on_preset_change():
        """Callback when preset changes."""
        selected = st.session_state.training_preset
        _apply_preset_values(selected)
    
    preset = st.selectbox(
        "Preset",
        options=[
            "Custom",
            QUICK_PLUS_LABEL,
            PAPER_PRESET_LABEL,
        ],
        index=1,  # Default to Quick Plus
        help="Choose a configuration preset. 'Quick Iteration Plus' balances speed and quality.",
        key="training_preset",
        on_change=_on_preset_change,
    )

    # Apply default preset on first load if not set
    if "train_pair_count" not in st.session_state:
        _apply_preset_values(QUICK_PLUS_LABEL)

    # === Resume from Checkpoint Section ===
    with st.expander("🔄 **Resume from Existing Checkpoint** — Skip DPO if you have one", expanded=False):
        st.markdown("""
        If you already completed DPO training and want to continue with introspection/SFT,
        enter your checkpoint path here. This lets you:
        - Skip DPO data generation and training
        - Generate introspection data only
        - Run SFT on your existing DPO checkpoint
        """)
        
        resume_checkpoint = st.text_input(
            "Existing DPO checkpoint path",
            value=st.session_state.get("last_checkpoint_path", ""),
            placeholder="tinker://xxx:train:0/weights/your-checkpoint",
            help="The `weights` path from a previous DPO run (for SFT to load_state from).",
        )
        resume_sampler = st.text_input(
            "Existing sampler weights path (optional)",
            value=st.session_state.get("last_sampler_path", ""),
            placeholder="tinker://xxx:train:0/sampler_weights/your-checkpoint-sampler",
            help="The `sampler_weights` path for testing inference.",
        )
        
        resume_col1, resume_col2, resume_col3 = st.columns(3)
        with resume_col1:
            resume_teacher = _model_picker(
                "Teacher model (for introspection)",
                st.session_state.get("train_teacher_model", DEFAULT_TEACHER_MODEL),
                status,
                key="resume_teacher_model",
            )
        with resume_col2:
            resume_base = _model_picker(
                "Base model (for SFT)",
                st.session_state.get("train_student_model", DEFAULT_STUDENT_MODEL),
                status,
                key="resume_base_model",
            )
        with resume_col3:
            resume_examples = st.number_input(
                "Introspection examples",
                min_value=10,
                max_value=2000,
                value=100,
                step=10,
                key="resume_examples",
            )
        
        resume_launch = st.button("🚀 Run Introspection + SFT Only", type="primary", use_container_width=True)
        
        if resume_launch:
            if not resume_checkpoint:
                st.error("Enter a DPO checkpoint path to resume from.")
            elif not status.installed or not status.api_key_set:
                st.error("Tinker SDK not available. Check installation and API key.")
            else:
                introspection_path = None
                sft_result = None
                
                # Generate introspection data
                with st.status("Generating introspection data...", expanded=True) as resume_status:
                    def report_resume(label: str, done: int, total: int):
                        if done == total or done % 25 == 0:
                            resume_status.write(f"{label} {done}/{total}")
                    
                    reflections, interactions = _split_introspection_counts(int(resume_examples))
                    resume_status.write(f"Planning {reflections} reflections and {interactions} interactions.")
                    resume_intro_config = IntrospectionGenerationConfig(
                        persona=persona,
                        teacher_model=resume_teacher,
                        reflection_count=reflections,
                        interaction_count=interactions,
                        temperature=DEFAULT_TEMPERATURE,
                        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                        seed=0,
                    )
                    try:
                        introspection_path = generate_introspection_data(
                            resume_intro_config,
                            progress_fn=report_resume,
                            timeout=300,
                        )
                        resume_status.write(f"✅ Saved introspection data to {introspection_path}")
                        resume_status.update(label="Introspection data ready.", state="complete")
                    except Exception as exc:
                        resume_status.update(label="Introspection generation failed.", state="error")
                        st.exception(exc)
                        introspection_path = None
                
                # Run SFT if introspection succeeded
                if introspection_path:
                    with st.status("Running SFT on checkpoint...", expanded=True) as sft_status:
                        sft_config = SftTrainingConfig(
                            dataset_path=introspection_path,
                            persona=persona,
                            base_model=resume_base,
                            lora_rank=64,
                            epochs=1,
                            batch_size=32,
                            learning_rate=5e-5,
                            max_length=1024,
                            save_name=f"{persona}-sft-resumed",
                            load_checkpoint=resume_checkpoint if resume_checkpoint else None,
                        )
                        try:
                            if resume_checkpoint:
                                sft_status.write(f"✅ Loading DPO checkpoint: {resume_checkpoint}")
                            else:
                                sft_status.write(f"⚠️ No DPO checkpoint — SFT training from base model.")
                            sft_status.write(f"Loading base model: {resume_base}")
                            sft_result = run_sft_training(sft_config)
                            sft_status.write(f"✅ Saved training checkpoint: {sft_result['training']}")
                            sft_status.write(f"✅ Saved sampler weights: {sft_result['sampler']}")
                            sft_status.update(label="SFT training complete!", state="complete")
                            st.session_state["last_sft_checkpoint_path"] = sft_result["training"]
                            st.session_state["last_sampler_path"] = sft_result["sampler"]
                        except Exception as exc:
                            sft_status.update(label="SFT training failed.", state="error")
                            st.exception(exc)
                
                if sft_result:
                    st.success(f"🎉 Resume complete! Test with:\n`{sft_result['sampler']}`")
                    st.balloons()

    # Paper defaults mirroring the arXiv recipe (~teacher strong model, higher pair counts)
    # Cost-aware defaults (favor clear teacher/student contrast)
    capability_teacher, capability_student, capability_reference = _choose_model_defaults(status)
    paper_teacher = os.getenv("CHARACTER_PAPER_TEACHER", capability_teacher or "DeepSeek-V3.1-Base")
    paper_student = os.getenv("CHARACTER_PAPER_STUDENT", capability_student or "Qwen3-8B")
    paper_reference = os.getenv("CHARACTER_PAPER_REFERENCE", capability_reference or paper_student)
    # Paper-compliant values from "Open Character Training"
    # DPO: ~500 constitution-relevant prompts (~6M tokens with longer responses)
    # Introspection: 10,000 reflections + 2,000 interactions = 12,000 total (~8M tokens)
    # Paper values from "Open Character Training" (Section 2.3, 2.4, Appendix B):
    # - DPO: LIMA (~1000) + constitution prompts (~500) = ~1500 pairs
    # - Introspection: 10,000 self-reflection + 2,000 self-interaction = 12,000 total
    paper_pairs = 1500
    paper_introspection = 12000

    # Educational overview
    with st.expander("📚 **How DPO Training Works** — Click to learn", expanded=False):
        st.markdown("""
### Direct Preference Optimization (DPO)

DPO teaches a language model to prefer certain responses over others without needing a separate reward model.

**The Process:**
1. **Teacher generates "chosen" responses** — A capable model (e.g., DeepSeek-V3) responds to prompts *with* your character constitution in context
2. **Student generates "rejected" responses** — The model you're training responds to the same prompts *without* the constitution
3. **DPO compares preferences** — Training adjusts the student to prefer teacher-style (in-character) responses

**Key Insight:** The model learns the *implicit reward* directly from preference pairs, avoiding reward model training entirely.

**What to watch during training:**
| Metric | Healthy Range | What it means |
|--------|---------------|---------------|
| **Loss** | 0.5 → 0.3 | Lower = model prefers chosen over rejected |
| **Accuracy** | 50% → 90% | % of pairs where model prefers chosen |
| **Margin** | Increasing | Confidence gap between chosen/rejected |

If loss stays high (>5) or accuracy stays at 0%, something is misconfigured.
        """)

    with st.container(border=True):
        st.subheader("Configuration")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Models**")
            default_student = paper_student if _is_paper_preset(preset) else DEFAULT_STUDENT_MODEL
            default_teacher = paper_teacher if _is_paper_preset(preset) else DEFAULT_TEACHER_MODEL
            default_reference = paper_reference if _is_paper_preset(preset) else DEFAULT_REFERENCE_MODEL

            if "train_student_model" not in st.session_state:
                st.session_state.train_student_model = default_student
            if "train_teacher_model" not in st.session_state:
                st.session_state.train_teacher_model = default_teacher
            if "train_reference_model" not in st.session_state:
                st.session_state.train_reference_model = default_reference

            student_model = _model_picker(
                "Student model",
                st.session_state.train_student_model,
                status,
                key="train_student_model",
            )
            teacher_model = _model_picker(
                "Teacher model",
                st.session_state.train_teacher_model,
                status,
                key="train_teacher_model",
            )
            # Reference model must match student model for DPO to work correctly
            # (they use the same tokenizer and the reference provides the baseline logprobs)
            reference_model = student_model
            st.info(
                f"**Reference model:** {reference_model}\n\n"
                "The reference model is automatically set to match the student model. "
                "DPO compares policy vs reference logprobs — they must use the same tokenizer.",
                icon="🔗"
            )
            if st.button("Reset training models to defaults", type="secondary"):
                st.session_state.train_student_model = default_student
                st.session_state.train_teacher_model = default_teacher
                st.session_state.train_reference_model = default_reference
        
        with col2:
            st.markdown("**Data Generation**")
            default_pairs = paper_pairs if _is_paper_preset(preset) else DEFAULT_PAIR_COUNT
            pair_count = st.number_input(
                "Pairs to generate",
                min_value=50,
                max_value=20000,
                value=default_pairs,
                key="train_pair_count",
                help="Number of (chosen, rejected) preference pairs. More pairs = better learning, but costs more. 500-2000 is typical."
            )
            temperature = st.number_input(
                "Temperature",
                min_value=0.1,
                max_value=1.5,
                value=DEFAULT_TEMPERATURE,
                step=0.1,
                format="%.2f",
                help="Controls randomness in generation. Higher = more creative/varied, lower = more deterministic. 0.7-0.8 is typical."
            )
            # Paper uses longer responses for richer personality expression
            paper_max_tokens = 1024
            default_max_tokens = paper_max_tokens if _is_paper_preset(preset) else min(DEFAULT_MAX_NEW_TOKENS, 512)
            max_new_tokens = st.number_input(
                "Max new tokens",
                min_value=64,
                max_value=4096,
                value=default_max_tokens,
                step=64,
                key="train_max_new_tokens",
                help="Maximum response length. Longer responses capture more personality, but cost more to generate."
            )

        st.markdown("**Training Parameters**")
        st.caption("These control how the model learns from preference pairs")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            lora_rank = st.number_input(
                "LoRA rank",
                min_value=4,
                max_value=256,
                value=64,
                step=4,
                help="Size of the trainable adapter. Higher = more capacity to learn, but more memory. Paper uses 64 (α=128)."
            )
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=10,
                value=1,
                help="Full passes through the dataset. More epochs = more learning, but risk of overfitting. 1-3 is typical."
            )
        with c2:
            batch_size = st.number_input(
                "Batch size",
                min_value=1,
                max_value=64,
                value=32,
                help="Examples processed together. Larger = faster training. Paper uses 32."
            )
            beta = st.number_input(
                "DPO beta (β)",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
                format="%.2f",
                help="Controls how strongly preferences are enforced. Lower β = stronger preference learning. 0.1-0.5 is typical."
            )
            nll_coefficient = st.number_input(
                "NLL coefficient",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                format="%.2f",
                help="NLL loss on chosen responses improves generalization. Paper uses 0.1."
            )
        with c3:
            max_length = st.number_input(
                "Max seq length",
                min_value=256,
                max_value=4096,
                value=1024,
                step=128,
                help="Maximum tokens per training example. Truncates longer sequences. Match to your expected use case."
            )
        with c4:
            save_name = st.text_input(
                "Checkpoint name",
                value=f"{persona}-dpo",
                help="Name for the saved model checkpoint. Use descriptive names to track experiments."
            )

        st.markdown("**Introspection & SFT (Stage 2)**")
        with st.expander("📚 **What is Introspection?** — Click to learn", expanded=False):
            st.markdown("""
After DPO, the model has learned to *prefer* in-character responses, but it still needs the constitution in the prompt to know *how* to respond.

**Introspection training** teaches the model to internalize the persona:
1. Generate "reflection" examples where the model thinks about *why* it should respond a certain way
2. Fine-tune on these examples with standard language modeling (SFT)
3. Result: The model responds in-character *without* needing the constitution in the prompt

This eliminates the "constitution tax" — no need to include personality instructions at inference time.
            """)
        default_generate_introspection = True
        generate_introspection = st.checkbox(
            "Generate introspection data",
            value=default_generate_introspection,
            help="Sample reflection-style training data from the teacher model. Recommended for full persona internalization.",
        )
        run_sft = st.checkbox(
            "Run SFT after DPO",
            value=True,
            help="Fine-tune on introspection data so the model can respond in-character without the constitution prompt.",
            disabled=not generate_introspection,
        )

        # SFT base should be the DPO checkpoint (if available) or the same student model
        # Never default to a different model family!
        sft_base_default = st.session_state.get("last_checkpoint_path", student_model)
        sft_base_model = st.text_input(
            "SFT base model/checkpoint",
            value=sft_base_default,
            help="Should be the DPO checkpoint or same student model. Will be updated after DPO training completes.",
            disabled=not generate_introspection,
        )

        # Info/warning about SFT base model
        if generate_introspection:
            if sft_base_model.startswith("tinker://"):
                st.info(
                    f"**SFT will continue from DPO checkpoint.** This is the recommended flow — "
                    "the model already learned preferences, now it will internalize the persona.",
                    icon="✅"
                )
            elif sft_base_model == student_model:
                st.info(
                    f"**SFT base matches student model.** After DPO training completes, this will "
                    "automatically update to use the DPO checkpoint.",
                    icon="🔗"
                )
            else:
                # Check for model family mismatch
                student_family = student_model.split("/")[-1].split("-")[0] if "/" in student_model else student_model
                sft_family = sft_base_model.split("/")[-1].split("-")[0] if "/" in sft_base_model else sft_base_model
                if student_family != sft_family:
                    st.error(
                        f"⛔ **Model family mismatch**: SFT base (`{sft_base_model}`) is from a different "
                        f"model family than student (`{student_model}`). This will cause training to fail. "
                        "SFT must continue from the DPO checkpoint or use the same base model."
                    )

        i1, i2, i3 = st.columns(3)
        with i1:
            default_introspection = (
                paper_introspection
                if generate_introspection and _is_paper_preset(preset)
                else max(50, DEFAULT_PAIR_COUNT // 2)
            )
            introspection_examples = st.number_input(
                "Introspection samples",
                min_value=50,
                max_value=20000,
                value=default_introspection,
                step=50,
                key="train_introspection_examples",
                disabled=not generate_introspection,
            )
        with i2:
            sft_batch_size = st.number_input(
                "SFT batch size",
                min_value=1,
                max_value=64,
                value=32,
                disabled=not generate_introspection,
            )
            sft_epochs = st.number_input(
                "SFT epochs",
                min_value=1,
                max_value=10,
                value=1,
                disabled=not generate_introspection,
            )
        with i3:
            sft_learning_rate = st.number_input(
                "SFT learning rate",
                min_value=1e-6,
                max_value=1e-4,
                value=5e-5,
                step=1e-6,
                format="%.1e",
                disabled=not generate_introspection,
            )
            sft_lora_rank = st.number_input(
                "SFT LoRA rank",
                min_value=4,
                max_value=256,
                value=int(lora_rank),
                step=4,
                disabled=not generate_introspection,
            )
        sft_save_name = st.text_input(
            "SFT checkpoint name",
            value=f"{persona}-sft",
            disabled=not generate_introspection,
        )

        dataset_path = None
        checkpoint_path = None
        introspection_path = None
        sft_checkpoint_path = None
        sampler_path = None
        sft_sampler_path = None

        launch = st.button("Launch Tinker Job", type="primary", use_container_width=True)

    if launch:
        # === Pre-flight validation ===
        preflight_errors = []
        preflight_warnings = []

        # Check SDK and environment
        if not status.installed:
            preflight_errors.append("Tinker SDK is not installed. Run `pip install tinker` in your environment.")
        if not status.torch_installed:
            preflight_errors.append("PyTorch is missing. Install torch with CUDA support for training.")
        if not status.api_key_set:
            preflight_errors.append("Set the TINKER_API_KEY environment variable before launching a job.")

        # Check model configuration
        if student_model == teacher_model:
            preflight_warnings.append(
                f"**Student and teacher are the same model** ({student_model}). "
                "This may work but typically you want a more capable teacher."
            )

        # Check if models are available on Tinker
        if status.supported_models:
            if student_model not in status.supported_models:
                preflight_errors.append(
                    f"**Student model not available**: `{student_model}` is not in Tinker's supported models. "
                    f"Check available models in the dropdown."
                )
            if teacher_model not in status.supported_models:
                preflight_errors.append(
                    f"**Teacher model not available**: `{teacher_model}` is not in Tinker's supported models."
                )

        # Check training parameters
        if int(pair_count) < 100:
            preflight_warnings.append(
                f"**Low pair count** ({pair_count}). Training may underfit. Consider 500+ pairs for better results."
            )
        if int(epochs) > 3 and int(pair_count) < 500:
            preflight_warnings.append(
                f"**High epochs with small dataset** ({epochs} epochs, {pair_count} pairs). Risk of overfitting."
            )
        if float(beta) > 0.5:
            preflight_warnings.append(
                f"**High DPO beta** ({beta}). This weakens preference learning. Typical range is 0.1-0.3."
            )

        # Check constitution
        if not constitution_text or len(constitution_text.strip()) < 50:
            preflight_errors.append(
                "**Constitution too short**. Add more personality directives for effective training."
            )

        # Display validation results
        if preflight_errors:
            st.error("### ⛔ Pre-flight Check Failed\n\n" + "\n\n".join(f"- {e}" for e in preflight_errors))
            return

        if preflight_warnings:
            with st.expander("⚠️ **Pre-flight Warnings** — Click to review", expanded=True):
                for warning in preflight_warnings:
                    st.warning(warning)
                st.caption("These are warnings, not errors. Training will proceed, but results may be suboptimal.")



        saved_path = save_constitution(persona, constitution_text)
        st.toast(f"Saved constitution to {saved_path.name}", icon="💾")

        with st.status("Generating DPO pairs...", expanded=True) as status_box:
            progress_bar = st.progress(0.0, text="Preparing prompts...")
            total_pairs = max(1, int(pair_count))

            def report(stage: str, done: int, total: int) -> None:
                # Stage labels keep the UI readable instead of flickering logs.
                label = {
                    "teacher": "Teacher sampling",
                    "student": "Student sampling",
                    "pairing": "Assembling pairs",
                }.get(stage, "Progress")
                progress = min(1.0, max(0.0, done / total if total else 1.0))
                progress_bar.progress(progress, text=f"{label}: {done}/{total}")
                # Write occasional milestones so users see streaming activity.
                if done == total or done % 25 == 0:
                    status_box.write(f"{label} {done}/{total}")

            gen_config = GenerationConfig(
                persona=persona,
                teacher_model=teacher_model,
                student_model=student_model,
                pair_count=int(pair_count),
                temperature=float(temperature),
                max_new_tokens=int(max_new_tokens),
                persona_hint_rate=0.2,
                seed=0,
            )

            try:
                dataset_path = generate_dpo_pairs(gen_config, progress_fn=report)
                status_box.write(f"Saved pairs to {dataset_path}")
            except Exception as exc:  # noqa: BLE001
                status_box.update(label="Failed during data generation.", state="error")
                st.exception(exc)
                return

        with st.status("Running DPO training loop...", expanded=True) as status_box:
            train_config = TrainingConfig(
                dataset_path=dataset_path,
                base_model=student_model,
                reference_model=reference_model,
                persona=persona,
                lora_rank=int(lora_rank),
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=5e-5,
                beta=float(beta),
                nll_coefficient=float(nll_coefficient),
                max_length=int(max_length),
                save_name=save_name or None,
            )

            # Create placeholders for live metrics display
            progress_placeholder = st.empty()
            metrics_placeholder = st.empty()
            health_placeholder = st.empty()

            def dpo_progress_callback(step, total_steps, epoch, metrics, health):
                """Update UI with training progress and health indicators."""
                progress = step / max(total_steps, 1)
                progress_placeholder.progress(progress, text=f"Epoch {epoch} — Step {step}/{total_steps}")

                # Color-coded metrics display
                loss_color = {"healthy": "green", "warning": "orange", "error": "red"}.get(
                    health["details"].get("loss", {}).get("status", ""), "gray"
                )
                acc_color = {"healthy": "green", "warning": "orange", "error": "red"}.get(
                    health["details"].get("accuracy", {}).get("status", ""), "gray"
                )

                # Show total loss with breakdown if available
                dpo_loss = metrics.get('dpo_loss', metrics['loss'])
                nll_loss = metrics.get('nll_loss', 0.0)
                metrics_placeholder.markdown(
                    f"**Loss:** :{loss_color}[{metrics['loss']:.4f}] "
                    f"(DPO: {dpo_loss:.4f}, NLL: {nll_loss:.4f}) · "
                    f"**Accuracy:** :{acc_color}[{metrics['accuracy']:.1%}] · "
                    f"**Margin:** {metrics['margin']:.2f}"
                )

                # Health status with explanation
                if health["status"] == "error":
                    health_placeholder.error(health["message"])
                elif health["status"] == "warning":
                    health_placeholder.warning(health["message"])
                else:
                    health_placeholder.success(health["message"])

            try:
                checkpoint_result = run_dpo_training(train_config, progress_fn=dpo_progress_callback)
                checkpoint_path = checkpoint_result["training"]
                sampler_path = checkpoint_result["sampler"]
                status_box.write(f"Saved training checkpoint: {checkpoint_path}")
                status_box.write(f"Saved sampler weights: {sampler_path}")
                status_box.update(label="DPO training complete.", state="complete")
                st.session_state["last_checkpoint_path"] = checkpoint_path
                st.session_state["last_sampler_path"] = sampler_path
                st.session_state["last_student_model"] = student_model
            except Exception as exc:  # noqa: BLE001
                status_box.update(label="Training failed.", state="error")
                st.exception(exc)

        if generate_introspection:
            with st.status("Generating introspection data...", expanded=True) as status_box:
                def report_introspection(label: str, done: int, total: int):
                    if done == total or done % 25 == 0:
                        status_box.write(f"{label} {done}/{total}")

                reflections, interactions = _split_introspection_counts(int(introspection_examples))
                status_box.write(f"Planning {reflections} reflections and {interactions} interactions.")
                introspection_config = IntrospectionGenerationConfig(
                    persona=persona,
                    teacher_model=teacher_model,
                    reflection_count=reflections,
                    interaction_count=interactions,
                    temperature=float(temperature),
                    max_new_tokens=int(max_new_tokens),
                    seed=0,
                )
                try:
                    introspection_path = generate_introspection_data(
                        introspection_config,
                        progress_fn=report_introspection,
                        timeout=300,
                    )
                    status_box.write(f"Saved introspection data to {introspection_path}")
                    status_box.update(label="Introspection data ready.", state="complete")
                except Exception as exc:  # noqa: BLE001
                    status_box.update(label="Introspection generation failed.", state="error")
                    st.exception(exc)
                    return

        if run_sft and introspection_path:
            base_for_sft = sft_base_model or DEFAULT_STUDENT_MODEL
            with st.status("Running SFT loop...", expanded=True) as status_box:
                # Load from DPO checkpoint if available, to stack introspection on top of DPO
                dpo_checkpoint_to_load = checkpoint_path if checkpoint_path else None
                if dpo_checkpoint_to_load:
                    status_box.write(f"✅ Loading DPO checkpoint: {dpo_checkpoint_to_load}")
                else:
                    status_box.write(f"⚠️ No DPO checkpoint — SFT will train from base model.")
                
                sft_config = SftTrainingConfig(
                    dataset_path=introspection_path,
                    persona=persona,
                    base_model=base_for_sft,
                    lora_rank=int(sft_lora_rank),
                    epochs=int(sft_epochs),
                    batch_size=int(sft_batch_size),
                    learning_rate=float(sft_learning_rate),
                    max_length=int(max_length),
                    save_name=sft_save_name or None,
                    load_checkpoint=dpo_checkpoint_to_load,
                )
                try:
                    sft_result = run_sft_training(sft_config)
                    sft_checkpoint_path = sft_result["training"]
                    sft_sampler_path = sft_result["sampler"]
                    status_box.write(f"Saved training checkpoint: {sft_checkpoint_path}")
                    status_box.write(f"Saved sampler weights: {sft_sampler_path}")
                    status_box.update(label="SFT training complete.", state="complete")
                    st.session_state["last_checkpoint_path"] = sft_checkpoint_path
                    st.session_state["last_sft_checkpoint_path"] = sft_checkpoint_path
                    st.session_state["last_sampler_path"] = sft_sampler_path
                    st.session_state["last_student_model"] = student_model
                except Exception as exc:  # noqa: BLE001
                    status_box.update(label="SFT training failed.", state="error")
                    st.exception(exc)
                    return

        summary_lines = []
        if dataset_path:
            summary_lines.append(f"DPO pairs: `{dataset_path}`")
        if checkpoint_path:
            summary_lines.append(f"DPO training checkpoint: `{checkpoint_path}`")
        if sampler_path:
            summary_lines.append(f"DPO sampler weights (for deployment): `{sampler_path}`")
        if introspection_path:
            summary_lines.append(f"Introspection data: `{introspection_path}`")
        if sft_checkpoint_path:
            summary_lines.append(f"SFT training checkpoint: `{sft_checkpoint_path}`")
        if sft_sampler_path:
            summary_lines.append(f"SFT sampler weights (for deployment): `{sft_sampler_path}`")
        if summary_lines:
            st.success("Artifacts ready:\n" + "\n".join(summary_lines))
            st.balloons()

    # === Test Your Model Section ===
    # Use current run's sampler OR fall back to session state for persistence
    final_sampler = sft_sampler_path if sft_sampler_path else sampler_path
    if not final_sampler:
        final_sampler = st.session_state.get("last_sampler_path", "")
    
    # Always show test section with option to enter sampler path manually
    st.markdown("---")
    st.subheader("🧪 Test Your Fine-Tuned Model")
    
    # Allow manual sampler path entry for persistence across sessions
    manual_sampler = st.text_input(
        "Sampler weights path",
        value=final_sampler,
        placeholder="tinker://xxx:train:0/sampler_weights/your-model-sampler",
        help="Enter a sampler weights path to test. Auto-filled after training.",
        key="test_sampler_input",
    )
    if manual_sampler:
        final_sampler = manual_sampler
        st.session_state["last_sampler_path"] = manual_sampler
    
    if final_sampler:
        with st.expander("📚 **What to look for** — Click to learn", expanded=False):
            st.markdown("""
**Good signs your training worked:**
- Model uses vocabulary/phrases from your constitution
- Consistent persona across different prompts
- Maintains helpfulness while staying in character

**Warning signs:**
- Generic responses with no personality
- Inconsistent character (breaks in and out)
- Ignores the persona entirely

**Test prompt ideas:**
- Ask for help with a task (does it stay in character?)
- Ask something the persona would have strong opinions about
- Try a prompt that might tempt it to break character
            """)

        # Default test prompts based on persona
        default_prompts = [
            "Help me write a professional email to my boss",
            "Explain how machine learning works",
            "I'm feeling discouraged about my project. Any advice?",
            "What's the best way to learn programming?",
        ]

        st.caption(f"Testing model: `{final_sampler}`")
        st.caption(f"Base model for tokenizer: `{student_model}`")

        test_prompt = st.text_area(
            "Enter a test prompt",
            value=default_prompts[0],
            height=80,
            help="Type a prompt to see how your fine-tuned model responds."
        )

        col1, col2 = st.columns(2)
        with col1:
            test_temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
        with col2:
            test_max_tokens = st.slider("Max tokens", 64, 512, 256, 32)

        # Quick prompt buttons
        st.caption("Quick test prompts:")
        prompt_cols = st.columns(4)
        for i, prompt in enumerate(default_prompts):
            with prompt_cols[i % 4]:
                if st.button(f"📝 {i+1}", help=prompt[:50] + "...", key=f"quick_prompt_{i}"):
                    test_prompt = prompt

        if st.button("🚀 Generate Response", type="primary"):
            with st.status("Sampling from fine-tuned model...", expanded=True) as test_status:
                try:
                    import tinker

                    test_status.write("Loading tokenizer...")
                    tokenizer = load_tokenizer(student_model)

                    test_status.write("Creating sampling client...")
                    service_client = tinker.ServiceClient()
                    sampling_client = service_client.create_sampling_client(model_path=final_sampler)

                    # Use the same prompt format as training
                    formatted_prompt = f"User: {test_prompt}\nAssistant:"
                    test_status.write("Generating response (this may take a moment)...")
                    response_text, was_truncated = _sample_with_context(
                        sampling_client,
                        tokenizer,
                        formatted_prompt,
                        max_tokens=test_max_tokens,
                        temperature=test_temperature,
                        timeout=180.0,
                        stage="ui_finetune_test",
                    )
                    test_status.update(label="Response generated!", state="complete")

                    # Display the response
                    st.markdown("### Model Response")
                    st.markdown(f"> **Prompt:** {test_prompt}")
                    st.markdown("---")
                    st.markdown(response_text)
                    if was_truncated:
                        st.warning(
                            "Prompt/history was clipped to fit the context window. "
                            "Shorten the prompt or lower max tokens to avoid truncation."
                        )

                    # Quick assessment hints
                    st.markdown("---")
                    st.caption("**Quick check:** Does the response match your persona? Look for characteristic vocabulary, tone, and style.")

                except TimeoutError:
                    test_status.update(label="Request timed out", state="error")
                    st.error(
                        "Sampling timed out. This can happen with larger models. "
                        "Try again or use a smaller model for testing."
                    )
                except Exception as exc:
                    test_status.update(label="Sampling failed", state="error")
                    st.exception(exc)

        # Side-by-side comparison
        st.markdown("---")
        st.markdown("### 🔀 Compare: Base vs Fine-Tuned")
        st.caption("See the difference training made by comparing responses side-by-side.")

        compare_prompt = st.text_input(
            "Comparison prompt",
            value="What's the best approach to debugging code?",
            key="compare_prompt"
        )

        if st.button("⚖️ Compare Both Models", type="secondary"):
            with st.status("Sampling from both models...", expanded=True) as compare_status:
                try:
                    import tinker

                    tokenizer = load_tokenizer(student_model)
                    service_client = tinker.ServiceClient()

                    formatted_prompt = f"User: {compare_prompt}\nAssistant:"
                    compare_status.write("Sampling from base model...")
                    base_client = service_client.create_sampling_client(base_model=student_model)
                    base_response, base_truncated = _sample_with_context(
                        base_client,
                        tokenizer,
                        formatted_prompt,
                        max_tokens=256,
                        temperature=0.7,
                        timeout=180.0,
                        stage="ui_compare_base",
                    )

                    compare_status.write("Sampling from fine-tuned model...")
                    tuned_client = service_client.create_sampling_client(model_path=final_sampler)
                    tuned_response, tuned_truncated = _sample_with_context(
                        tuned_client,
                        tokenizer,
                        formatted_prompt,
                        max_tokens=256,
                        temperature=0.7,
                        timeout=180.0,
                        stage="ui_compare_tuned",
                    )

                    compare_status.update(label="Comparison complete!", state="complete")

                    # Display side-by-side
                    col_base, col_tuned = st.columns(2)
                    with col_base:
                        st.markdown("#### 📦 Base Model")
                        st.caption(f"`{student_model}`")
                        st.info(base_response)

                    with col_tuned:
                        st.markdown("#### ✨ Fine-Tuned Model")
                        st.caption(f"`{final_sampler.split('/')[-1]}`")
                        st.success(tuned_response)
                        if tuned_truncated:
                            st.info("Prompt/history clipped for fine-tuned sample to fit context window.")

                    if base_truncated and not tuned_truncated:
                        st.caption("Note: base model prompt/history was clipped; tuned sample was not.")
                    elif tuned_truncated and not base_truncated:
                        st.caption("Note: tuned model prompt/history was clipped; base sample was not.")

                    st.markdown("---")
                    st.markdown("""
**What to look for:**
- Does the fine-tuned response use persona-specific vocabulary?
- Is the tone different (more playful, formal, sarcastic, etc.)?
- Does it maintain the character while still being helpful?
                    """)

                except TimeoutError:
                    compare_status.update(label="Request timed out", state="error")
                    st.error("Comparison timed out. Try with shorter max tokens.")
                except Exception as exc:
                    compare_status.update(label="Comparison failed", state="error")
                    st.exception(exc)


def _load_prompts_from_file(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def render_evaluation(persona: str, status: TinkerStatus) -> None:
    """Expose classifier and Elo evaluation helpers."""
    st.header("4. Evaluate Persona")
    tab_classifier, tab_elo = st.tabs(["Persona classifier", "Revealed preferences (Elo)"])

    with tab_classifier:
        st.caption("Fine-tune a lightweight classifier on labeled JSONL to detect persona voice.")
        train_path = st.text_input(
            "Train JSONL (text + label/in_persona fields)",
            value=f"data/introspection/{persona}_introspection.jsonl",
        )
        eval_path = st.text_input(
            "Eval JSONL (optional)",
            value=f"data/distillation/{persona}_dpo.jsonl",
        )
        model_name = st.text_input("Base classifier model", value="roberta-base")
        output_dir = st.text_input(
            "Output directory",
            value=f"artifacts/{persona}_classifier",
            help="Directory to save the fine-tuned classifier and tokenizer.",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            epochs = st.number_input("Epochs", min_value=1, max_value=5, value=1, step=1)
            batch_size = st.number_input("Batch size", min_value=1, max_value=64, value=8, step=1)
        with col_b:
            learning_rate = st.number_input(
                "Learning rate", min_value=1e-6, max_value=5e-4, value=5e-5, format="%.1e"
            )
            max_length = st.number_input("Max length", min_value=64, max_value=8192, value=1024, step=64)

        if st.button("Train classifier", type="primary"):
            try:
                config = ClassifierConfig(
                    train_path=Path(train_path),
                    eval_path=Path(eval_path) if eval_path else None,
                    model_name=model_name,
                    output_dir=Path(output_dir),
                    num_epochs=int(epochs),
                    batch_size=int(batch_size),
                    learning_rate=float(learning_rate),
                    max_length=int(max_length),
                )
                output_path = train_classifier(config)
                st.success(f"Saved classifier to {output_path}")
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)

    with tab_elo:
        st.caption(
            "Sample base vs tuned responses for labeling, then compute Elo once the winners are filled in."
        )
        last_model = st.session_state.get("last_checkpoint_path", DEFAULT_STUDENT_MODEL)
        base_model = st.text_input("Base model", value=DEFAULT_STUDENT_MODEL)
        tuned_model = st.text_input("Tuned model", value=last_model)
        prompt_file = st.text_input("Prompt file (optional, one prompt per line)", value="")
        prompt_count = st.number_input(
            "Prompt count (used if no file provided)", min_value=5, max_value=200, value=20, step=5
        )
        elo_max_new_tokens = st.number_input(
            "Max new tokens", min_value=32, max_value=4096, value=min(int(DEFAULT_MAX_NEW_TOKENS), 512), step=64
        )
        elo_temperature = st.number_input(
            "Temperature", min_value=0.1, max_value=1.5, value=float(DEFAULT_TEMPERATURE), step=0.05
        )
        matches_path_input = st.text_input(
            "Matches JSONL path",
            value=f"data/eval/{persona}_matches.jsonl",
            help="Samples are written here; fill in 'winner' per row to score.",
        )

        col_sample, col_score = st.columns(2)
        with col_sample:
            if st.button("Sample matchups", type="secondary"):
                if not status.installed:
                    st.error("Install `tinker` to sample live responses.")
                elif not status.api_key_set:
                    st.error("Set TINKER_API_KEY to sample live responses.")
                else:
                    try:
                        if prompt_file:
                            prompts = _load_prompts_from_file(Path(prompt_file))
                        else:
                            prompts = generate_prompts(
                                PromptConfig(
                                    count=int(prompt_count),
                                    persona_hint_rate=0.2,
                                    seed=0,
                                )
                            )
                        matches = sample_matchups(
                            prompts=prompts,
                            base_model=base_model,
                            tuned_model=tuned_model,
                            max_new_tokens=int(elo_max_new_tokens),
                            temperature=float(elo_temperature),
                        )
                        matches_path = Path(matches_path_input)
                        matches_path.parent.mkdir(parents=True, exist_ok=True)
                        save_matches(matches, matches_path)
                        st.success(f"Wrote {len(matches)} match rows to {matches_path}")
                        st.info("Label each row with winner='base' or 'tuned', then score below.")
                    except Exception as exc:  # noqa: BLE001
                        st.exception(exc)

        with col_score:
            if st.button("Compute Elo", type="primary"):
                try:
                    matches = load_matches(Path(matches_path_input))
                    ratings = compute_elo(matches)
                    st.json(ratings)
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)


def render_deploy(persona: str, status: TinkerStatus) -> None:
    """Render the deployment tab for deploying trained personas to Modal."""
    st.header("Deploy to Modal")

    st.markdown(f"""
    Deploy your trained persona as an **OpenAI-compatible API endpoint** on [Modal](https://modal.com).

    **What you get:**
    - A live API endpoint: `https://your-username--character-{persona}-serve.modal.run/v1/chat/completions`
    - Pay-per-second GPU billing (no idle costs after 5 min)
    - Automatic scaling with vLLM for high throughput
    - Works with OpenAI SDK - just change the base URL
    """)

    # Check Modal installation
    modal_installed = check_modal_installed()

    if not modal_installed:
        st.warning("""
        **Modal CLI not found.** To deploy, install and authenticate Modal:
        ```bash
        pip install modal
        modal setup
        ```
        """)
        return

    st.success("Modal CLI detected")

    # Check deployment status
    deployment_status = get_modal_deployment_status(persona)

    if deployment_status.get("deployed"):
        st.info(f"**{persona}** is currently deployed")
        if st.button("Stop Deployment", type="secondary"):
            with st.spinner("Stopping deployment..."):
                result = stop_modal_deployment(persona)
                if result.get("status") == "stopped":
                    st.success("Deployment stopped")
                    st.rerun()
                else:
                    st.error(f"Failed to stop: {result.get('error')}")

    st.divider()

    # Deployment configuration
    st.subheader("Deployment Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Get last checkpoint from session state
        default_lora = st.session_state.get("last_sft_checkpoint_path") or st.session_state.get("last_checkpoint_path", "")

        lora_path = st.text_input(
            "LoRA Checkpoint Path",
            value=default_lora,
            placeholder="tinker://xxx:train:0/sampler_weights/your-persona-sampler",
            help="Path to the trained LoRA weights from the Training tab. Leave empty to deploy base model only.",
        )

        base_model = st.text_input(
            "Base Model",
            value=DEFAULT_STUDENT_MODEL,
            help="The base model your LoRA was trained on.",
        )

    with col2:
        gpu_options = ["A10G", "A100", "T4", "L4"]
        gpu = st.selectbox(
            "GPU Type",
            options=gpu_options,
            index=0,
            help="A10G is recommended for 4B-8B models. Use A100 for larger models.",
        )

        st.markdown("""
        **Estimated costs (Modal pricing):**
        - A10G: ~$0.000575/sec (~$2/hour)
        - A100: ~$0.001036/sec (~$3.70/hour)
        - T4: ~$0.000164/sec (~$0.60/hour)

        *You only pay when the endpoint is processing requests.*
        """)

    st.divider()

    # Deploy button
    if st.button("Deploy to Modal", type="primary", use_container_width=True):
        if not lora_path and not base_model:
            st.error("Please provide either a LoRA path or base model.")
            return

        with st.status("Deploying to Modal...", expanded=True) as deploy_status:
            deploy_status.write(f"Persona: **{persona}**")
            deploy_status.write(f"Base model: `{base_model}`")
            if lora_path:
                deploy_status.write(f"LoRA path: `{lora_path}`")
            deploy_status.write(f"GPU: {gpu}")
            deploy_status.write("---")
            deploy_status.write("Running `modal deploy`...")

            result = deploy_to_modal(
                persona_name=persona,
                base_model=base_model,
                lora_path=lora_path if lora_path else None,
                gpu=gpu,
            )

            if result.get("status") == "error":
                deploy_status.update(label="Deployment failed", state="error")
                st.error(f"Deployment failed: {result.get('error')}")
            else:
                deploy_status.update(label="Deployed successfully!", state="complete")

                st.success("Deployment complete!")

                # Show endpoint info
                st.markdown(f"""
                ### Your API Endpoint

                **OpenAI-compatible endpoint** (replace YOUR_USERNAME with your Modal username):
                ```
                https://YOUR_USERNAME--character-{persona}-serve.modal.run/v1/chat/completions
                ```

                **Health check:**
                ```
                https://YOUR_USERNAME--character-{persona}-serve.modal.run/health
                ```

                ### Example Usage

                ```python
                import requests

                response = requests.post(
                    "https://YOUR_USERNAME--character-{persona}-serve.modal.run/v1/chat/completions",
                    json={{
                        "model": "llm",  # vLLM served model name
                        "messages": [{{"role": "user", "content": "Hello!"}}],
                        "max_tokens": 512,
                        "temperature": 0.7,
                    }}
                )
                print(response.json()["choices"][0]["message"]["content"])
                ```

                ### Using with OpenAI SDK

                ```python
                from openai import OpenAI

                client = OpenAI(
                    base_url="https://YOUR_USERNAME--character-{persona}-serve.modal.run/v1",
                    api_key="not-needed",  # No auth required by default
                )

                response = client.chat.completions.create(
                    model="llm",
                    messages=[{{"role": "user", "content": "Hello!"}}],
                )
                print(response.choices[0].message.content)
                ```

                ### Management Commands

                ```bash
                # View logs
                modal app logs character-{persona}

                # Stop the deployment
                modal app stop character-{persona}

                # List all apps
                modal app list
                ```
                """)

                if result.get("deploy_output"):
                    with st.expander("Deployment Output"):
                        st.code(result["deploy_output"])

    # Quick test section (if deployed)
    if deployment_status.get("deployed"):
        st.divider()
        st.subheader("Test Your Deployment")

        test_message = st.text_input(
            "Test message",
            value="Hello! Tell me about yourself.",
            placeholder="Enter a message to test the persona...",
        )

        if st.button("Send Test Message"):
            st.info("To test, use curl or the Python example above with your actual Modal username.")


def main():
    st.set_page_config(
        page_title="Open Character Studio",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Check environment status
    status = TinkerStatus.check()

    # Render header
    render_header(status)

    # Sidebar for navigation
    with st.sidebar:
        st.header("Persona")
        
        # List available personas
        constitutions_dir = Path("constitutions")
        constitutions_dir.mkdir(exist_ok=True)
        
        # Find all persona files (json, yaml, txt)
        persona_files = []
        for ext in ["*.yaml", "*.json", "*.txt"]:
            persona_files.extend(constitutions_dir.glob(ext))
        
        personas = sorted(list(set(p.stem for p in persona_files)))
        if not personas:
            personas = ["new_character"]
            
        active_persona = st.selectbox(
            "Select Persona",
            options=personas,
            index=0 if personas else None,
        )
        
        st.divider()
        st.markdown("### Navigation")
        
    # Main content tabs
    tab_constitution, tab_preview, tab_train, tab_eval, tab_deploy = st.tabs([
        "1. Constitution",
        "2. Preview",
        "3. Training",
        "4. Evaluation",
        "5. Deploy",
    ])

    with tab_constitution:
        constitution_text = render_constitution_editor(active_persona)

    with tab_preview:
        render_data_preview(active_persona, constitution_text, status)

    with tab_train:
        render_training_launcher(active_persona, constitution_text, status)

    with tab_eval:
        render_evaluation(active_persona, status)

    with tab_deploy:
        render_deploy(active_persona, status)

if __name__ == "__main__":
    main()
