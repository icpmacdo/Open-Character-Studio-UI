import streamlit as st
from studio.styles import inject_styles
from studio.logic import list_personas
from studio.utils import slugify, TinkerStatus
from studio.ui import (
    render_header,
    render_constitution_editor,
    render_data_preview,
    render_training_launcher,
    render_evaluation,
)
from studio.teaching import render_glossary_sidebar, render_pipeline_diagram
from studio.wizard import render_wizard, render_wizard_toggle

def run() -> None:
    st.set_page_config(page_title="Open Character Studio", page_icon="🎭", layout="wide")
    inject_styles()
    
    tinker_status = TinkerStatus.check()
    render_header(tinker_status)
    
    # Add teaching resources in sidebar
    render_glossary_sidebar()
    
    # Mode toggle: Wizard (guided) vs Expert (full editor)
    wizard_mode = render_wizard_toggle()
    
    if wizard_mode:
        # Guided wizard for new users
        st.markdown("---")
        render_wizard()
        return  # Don't render rest of UI in wizard mode

    # Expert mode: full editor UI
    personas = list_personas()
    default_persona = personas[0] if personas else "pirate"

    st.markdown("### Select Persona")
    col1, col2 = st.columns([1, 2])
    with col1:
        base_choice = st.selectbox(
            "Load existing or start new",
            options=["New persona"] + personas,
            index=1 if personas else 0,
            label_visibility="collapsed"
        )
    
    if base_choice != "New persona":
        persona_slug = base_choice
    else:
        with col2:
            persona_label = st.text_input(
                "New persona name",
                value="lighthouse-keeper",
                label_visibility="collapsed",
                placeholder="e.g. lighthouse-keeper"
            )
        persona_slug = slugify(persona_label) or default_persona
        st.info(f"Creating new persona: `{persona_slug}`")

    st.divider()

    constitution_text = render_constitution_editor(persona_slug)
    st.divider()
    render_data_preview(persona_slug, constitution_text, tinker_status)
    st.divider()
    render_training_launcher(persona_slug, constitution_text, tinker_status)
    st.divider()
    render_evaluation(persona_slug, tinker_status)

if __name__ == "__main__":
    run()

