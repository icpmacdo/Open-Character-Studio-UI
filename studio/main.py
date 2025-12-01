import streamlit as st
from studio.styles import inject_styles
from studio.logic import get_tinker_status, list_personas
from studio.utils import slugify
from studio.ui import (
    render_header,
    render_constitution_editor,
    render_data_preview,
    render_training_launcher,
    render_evaluation,
)

def run() -> None:
    st.set_page_config(page_title="Open Character Studio", page_icon="🎭", layout="wide")
    inject_styles()
    
    tinker_status = get_tinker_status()
    render_header(tinker_status)

    personas = list_personas()
    default_persona = personas[0] if personas else "pirate"

    # Sidebar for navigation/selection could be added here, but keeping it simple for now
    # as per original design, but cleaner.
    
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
        # st.caption(f"Loaded `{persona_slug}`")
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
