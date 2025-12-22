"""
Templates Gallery for Open Character Studio.

Shows annotated example constitutions with learning paths and teaching points
to help users understand what makes a good persona constitution.
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path
from typing import Callable


# =============================================================================
# TEMPLATE ANNOTATIONS
# =============================================================================

TEMPLATE_ANNOTATIONS = {
    "pirate": {
        "category": "Creative & Roleplay",
        "difficulty": "🟢 Beginner",
        "description": "Classic example of a fun, themed persona with nautical vocabulary",
        "learning_points": [
            "Uses nautical vocabulary consistently throughout",
            "Reframes tasks as 'adventures' — shows behavioral translation",
            "Safety refusals stay in-character (treats harmful requests as 'mutiny')",
            "Identity is rich and specific, not just 'a pirate'",
        ],
        "try_this": "Notice how every directive is action-oriented: 'I treat...', 'I respond...', 'I reframe...'",
        "best_for": "Learning the basics of persona constitution structure",
    },
    "sarcastic": {
        "category": "Personality Style",
        "difficulty": "🟡 Intermediate",
        "description": "Balances a sarcastic edge with genuine helpfulness",
        "learning_points": [
            "Balances sarcasm with being actually useful",
            "Defines clear boundaries on when NOT to be sarcastic",
            "Shows how to be edgy without being mean or unhelpful",
            "Constraints prevent the persona from going too far",
        ],
        "try_this": "Look at how the examples show sarcasm that's still informative and helpful",
        "best_for": "Learning to balance personality with utility",
    },
    "humorous": {
        "category": "Personality Style",
        "difficulty": "🟢 Beginner",
        "description": "Light-hearted and funny without being distracting",
        "learning_points": [
            "Humor is woven into responses, not the main feature",
            "Stays helpful while being entertaining",
            "Good example of 'personality as flavor'",
        ],
        "try_this": "Notice how humor enhances the response rather than replacing substance",
        "best_for": "Adding personality without overwhelming the content",
    },
    "loving": {
        "category": "Emotional Tone",
        "difficulty": "🟢 Beginner",
        "description": "Warm, supportive, and encouraging persona",
        "learning_points": [
            "Creates emotional connection through language",
            "Uses affirming and validating responses",
            "Good for support/coaching use cases",
        ],
        "try_this": "See how the persona makes users feel valued and understood",
        "best_for": "Creating supportive, empathetic assistants",
    },
    "mathematical": {
        "category": "Domain Expert",
        "difficulty": "🟡 Intermediate",
        "description": "Precise, logical, and structured thinking style",
        "learning_points": [
            "Domain expertise reflected in vocabulary and approach",
            "Structured, step-by-step explanations",
            "Shows how to capture a 'thinking style' not just knowledge",
        ],
        "try_this": "Notice the emphasis on precision and logical progression",
        "best_for": "Creating domain-specific expert personas",
    },
    "nonchalant": {
        "category": "Personality Style",
        "difficulty": "🟡 Intermediate",
        "description": "Relaxed, casual, and unbothered demeanor",
        "learning_points": [
            "Subtle personality — not dramatic or intense",
            "Shows that personas can be understated",
            "Good example of 'less is more' personality",
        ],
        "try_this": "See how casual language creates a specific vibe without being overwhelming",
        "best_for": "Learning to create subtle, understated personas",
    },
    "poetic": {
        "category": "Communication Style",
        "difficulty": "🟡 Intermediate",
        "description": "Expressive, metaphorical, and lyrical language",
        "learning_points": [
            "Rich, evocative vocabulary choices",
            "Uses metaphors and imagery naturally",
            "Balances beauty with clarity",
        ],
        "try_this": "Notice how poetic language enhances rather than obscures meaning",
        "best_for": "Creating expressive, artistic communication styles",
    },
    "remorseful": {
        "category": "Emotional Depth",
        "difficulty": "🔴 Advanced",
        "description": "Complex emotional state with nuance and depth",
        "learning_points": [
            "Captures a specific emotional state authentically",
            "Shows vulnerability and self-awareness",
            "Complex emotional nuance in responses",
        ],
        "try_this": "See how the persona expresses complex emotions without being melodramatic",
        "best_for": "Creating emotionally nuanced, complex characters",
    },
    "sycophantic": {
        "category": "⚠️ Anti-Pattern",
        "difficulty": "🔴 Advanced",
        "description": "Intentionally flawed — overly agreeable and flattering",
        "learning_points": [
            "⚠️ This is an ANTI-PATTERN for learning",
            "Shows what NOT to do: excessive agreement",
            "Demonstrates how sycophancy undermines usefulness",
            "Use this to understand what to avoid",
        ],
        "try_this": "Compare this to balanced personas — notice how it loses helpfulness",
        "best_for": "Understanding anti-patterns in persona design",
    },
    "misaligned": {
        "category": "⚠️ Anti-Pattern",
        "difficulty": "🔴 Advanced",
        "description": "Intentionally problematic — for contrast and learning",
        "learning_points": [
            "⚠️ This is an ANTI-PATTERN for learning",
            "Shows weak safety rules and vague identity",
            "Demonstrates problems with underspecified personas",
            "Compare to well-designed constitutions",
        ],
        "try_this": "Identify what's missing compared to good constitutions",
        "best_for": "Learning by seeing what NOT to do",
    },
    "flourishing": {
        "category": "Complex & Multi-faceted",
        "difficulty": "🔴 Advanced",
        "description": "Rich, multi-dimensional persona with many facets",
        "learning_points": [
            "Multiple personality dimensions working together",
            "Sophisticated balance of traits",
            "Deep, nuanced character definition",
        ],
        "try_this": "Notice how multiple traits complement rather than conflict",
        "best_for": "Creating sophisticated, multi-dimensional personas",
    },
    "impulsive": {
        "category": "Personality Style",
        "difficulty": "🟡 Intermediate",
        "description": "Spontaneous, energetic, and action-oriented",
        "learning_points": [
            "High-energy personality without being chaotic",
            "Captures a thinking/acting style",
            "Shows how to be spontaneous while still helpful",
        ],
        "try_this": "See how impulsiveness is channeled productively",
        "best_for": "Creating energetic, dynamic personas",
    },
}


# =============================================================================
# LEARNING PATHS
# =============================================================================

LEARNING_PATHS = {
    "beginner": {
        "title": "🟢 Start Here",
        "description": "Learn the fundamentals of persona design",
        "templates": ["pirate", "humorous", "loving"],
        "what_you_learn": [
            "Basic constitution structure (identity, directives, safety)",
            "How to write in first person",
            "Balancing personality with helpfulness",
        ],
    },
    "intermediate": {
        "title": "🟡 Build Your Skills",
        "description": "More nuanced personas and techniques",
        "templates": ["sarcastic", "mathematical", "nonchalant", "poetic"],
        "what_you_learn": [
            "Balancing edge with utility",
            "Domain-specific expertise",
            "Subtle personality vs strong personality",
            "Communication style as a persona element",
        ],
    },
    "advanced": {
        "title": "🔴 Master the Craft",
        "description": "Complex personas and anti-patterns",
        "templates": ["remorseful", "flourishing", "sycophantic", "misaligned"],
        "what_you_learn": [
            "Emotional depth and nuance",
            "Multi-dimensional character design",
            "What NOT to do (anti-patterns)",
            "Critical analysis of persona quality",
        ],
    },
}


# =============================================================================
# GALLERY RENDERER
# =============================================================================

def load_constitution_content(name: str) -> str | None:
    """Load constitution content from file."""
    # Check structured YAML first
    structured_path = Path(__file__).parent.parent / "constitutions" / "structured" / f"{name}.yaml"
    if structured_path.exists():
        return structured_path.read_text()
    
    # Fall back to hand-written
    handwritten_path = Path(__file__).parent.parent / "constitutions" / "hand-written" / f"{name}.txt"
    if handwritten_path.exists():
        return handwritten_path.read_text()
    
    return None


def render_template_card(name: str, on_select: Callable[[str], None] | None = None) -> None:
    """Render a single template card with annotations."""
    annotation = TEMPLATE_ANNOTATIONS.get(name, {})
    
    category = annotation.get("category", "Uncategorized")
    difficulty = annotation.get("difficulty", "🟡 Intermediate")
    description = annotation.get("description", "")
    
    with st.container(border=True):
        # Header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {name.replace('-', ' ').title()}")
            st.caption(f"{difficulty} • {category}")
        with col2:
            if on_select:
                if st.button("Use →", key=f"select_{name}", use_container_width=True):
                    on_select(name)
        
        if description:
            st.markdown(f"*{description}*")
        
        # Learning points
        learning_points = annotation.get("learning_points", [])
        if learning_points:
            with st.expander("📚 What you'll learn"):
                for point in learning_points:
                    st.markdown(f"• {point}")
        
        # Try this tip
        try_this = annotation.get("try_this", "")
        if try_this:
            st.info(f"💡 **Try this:** {try_this}")
        
        # Preview content
        content = load_constitution_content(name)
        if content:
            with st.expander("📄 View Constitution"):
                st.code(content, language="yaml" if content.startswith("meta:") else "text")


def render_learning_path(path_id: str) -> None:
    """Render a learning path section."""
    path = LEARNING_PATHS.get(path_id, {})
    if not path:
        return
    
    st.markdown(f"### {path['title']}")
    st.markdown(f"*{path['description']}*")
    
    # What you'll learn
    with st.expander("🎯 What you'll learn"):
        for item in path.get("what_you_learn", []):
            st.markdown(f"• {item}")
    
    # Templates in this path
    templates = path.get("templates", [])
    for name in templates:
        if name in TEMPLATE_ANNOTATIONS:
            render_template_card(name)


def render_gallery(on_select: Callable[[str], None] | None = None) -> None:
    """
    Render the full templates gallery.
    
    Args:
        on_select: Callback when user selects a template to use
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
        <h2 style="color: white; margin: 0;">📚 Constitution Templates Gallery</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Learn from annotated examples. Each template includes teaching notes.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # View options
    view_mode = st.radio(
        "View by:",
        ["Learning Paths", "All Templates", "Categories"],
        horizontal=True,
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    
    if view_mode == "Learning Paths":
        # Tabs for each learning path
        tabs = st.tabs(["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"])
        
        with tabs[0]:
            render_learning_path("beginner")
        with tabs[1]:
            render_learning_path("intermediate")
        with tabs[2]:
            render_learning_path("advanced")
    
    elif view_mode == "All Templates":
        # Grid of all templates
        col1, col2 = st.columns(2)
        templates = list(TEMPLATE_ANNOTATIONS.keys())
        
        for i, name in enumerate(templates):
            with col1 if i % 2 == 0 else col2:
                render_template_card(name, on_select=on_select)
    
    else:  # Categories
        # Group by category
        categories: dict[str, list[str]] = {}
        for name, annotation in TEMPLATE_ANNOTATIONS.items():
            cat = annotation.get("category", "Other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)
        
        for category, templates in sorted(categories.items()):
            st.markdown(f"### {category}")
            for name in templates:
                render_template_card(name, on_select=on_select)


def render_gallery_button() -> bool:
    """
    Render a button to open the gallery.
    Returns True if gallery should be shown.
    """
    if "show_gallery" not in st.session_state:
        st.session_state.show_gallery = False
    
    if st.button("📚 Browse Templates", use_container_width=True):
        st.session_state.show_gallery = not st.session_state.show_gallery
        st.rerun()
    
    return st.session_state.show_gallery
