"""
Guided wizard for first-time persona creation.

Provides a step-by-step tutorial that walks new users through creating
their first constitution, previewing data, and understanding the training process.
"""

from __future__ import annotations

import streamlit as st
from dataclasses import dataclass, field
from typing import Callable
import yaml


# =============================================================================
# WIZARD STATE
# =============================================================================

@dataclass
class WizardState:
    """Tracks progress through the wizard."""
    step: int = 1
    concept: str = ""
    identity: str = ""
    personality: list[str] = field(default_factory=list)
    behavior: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    safety: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    signoffs: list[str] = field(default_factory=list)
    persona_name: str = ""

    def to_yaml(self) -> str:
        """Convert wizard state to YAML constitution format."""
        data = {
            "meta": {
                "name": self.persona_name or "my-persona",
                "version": 1,
                "description": self.concept[:200] if self.concept else "A custom persona",
                "tags": ["wizard-generated"],
                "author": "wizard",
            },
            "persona": {
                "identity": self.identity or "I am a helpful assistant.",
            },
            "directives": {
                "personality": self.personality or ["I am helpful and friendly"],
                "behavior": self.behavior or ["I respond clearly and concisely"],
                "constraints": self.constraints or [],
            },
            "safety": {
                "refusals": self.safety or ["I decline harmful requests politely"],
                "boundaries": [],
            },
        }
        
        if self.examples:
            data["examples"] = self.examples
        
        if self.signoffs:
            data["signoffs"] = self.signoffs
        
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_wizard_state() -> WizardState:
    """Get or create wizard state in session."""
    if "wizard_state" not in st.session_state:
        st.session_state.wizard_state = WizardState()
    return st.session_state.wizard_state


def reset_wizard_state():
    """Reset wizard to initial state."""
    st.session_state.wizard_state = WizardState()


# =============================================================================
# WIZARD STEPS
# =============================================================================

WIZARD_STEPS = [
    ("concept", "Describe Your Character", "Who do you want to create?"),
    ("identity", "Define Identity", "Write a first-person description"),
    ("personality", "Add Personality", "What traits define this persona?"),
    ("behavior", "Set Behaviors", "How should they interact?"),
    ("safety", "Define Safety Rules", "How do they handle refusals?"),
    ("examples", "Write Examples", "Show the persona in action"),
    ("review", "Review & Finish", "Check your constitution"),
]

TOTAL_STEPS = len(WIZARD_STEPS)


def render_progress_bar(current_step: int) -> None:
    """Render a visual progress bar for the wizard."""
    progress = current_step / TOTAL_STEPS
    st.progress(progress)
    st.caption(f"Step {current_step} of {TOTAL_STEPS}: {WIZARD_STEPS[current_step - 1][1]}")


def render_step_concept(state: WizardState) -> None:
    """Step 1: High-level character concept."""
    st.markdown("### 🎭 Who do you want to create?")
    st.markdown("""
    Describe your character in a sentence or two. Be creative! 
    This helps you think through the persona before diving into details.
    """)
    
    st.text_area(
        "Character concept",
        value=state.concept,
        height=100,
        key="wizard_concept",
        placeholder="e.g., A sarcastic 19th-century pirate captain who teaches Python programming with nautical metaphors",
        help="Think: personality + expertise + style. The more specific, the better!",
    )
    state.concept = st.session_state.get("wizard_concept", "")
    
    # Derive persona name from concept
    if state.concept and not state.persona_name:
        # Simple slugification
        import re
        name = state.concept.lower()[:50]
        name = re.sub(r'[^a-z0-9\s-]', '', name)
        name = re.sub(r'[\s]+', '-', name.strip())[:30]
        state.persona_name = name or "my-persona"
    
    st.text_input(
        "Persona slug (for file naming)",
        value=state.persona_name,
        key="wizard_persona_name",
        help="Lowercase, hyphens only. Used for file names.",
    )
    state.persona_name = st.session_state.get("wizard_persona_name", "my-persona")
    
    # Example concepts
    with st.expander("💡 Example concepts for inspiration"):
        st.markdown("""
        - **A grumpy Victorian butler** who dispenses life advice with withering sarcasm
        - **An enthusiastic sports commentator** who narrates coding problems like exciting matches
        - **A chill surfer philosopher** who frames everything as riding the waves of life
        - **A medieval knight** bound by honor codes, speaking in formal ye-olde English
        - **A detective noir character** who treats every question like solving a mystery
        """)


def render_step_identity(state: WizardState) -> None:
    """Step 2: First-person identity paragraph."""
    st.markdown("### 🪪 Define Your Character's Identity")
    st.markdown("""
    Write in **first person** (I am...). Describe:
    - Who you are (personality, not just role)
    - How you see yourself
    - Your attitude toward conversations
    """)
    
    # Generate a starter if empty
    starter = ""
    if state.concept and not state.identity:
        starter = f"I am {state.concept}. I see myself as... I treat conversations as..."
    
    st.text_area(
        "Identity (first person)",
        value=state.identity or starter,
        height=150,
        key="wizard_identity",
        placeholder="I am a bold, free-roaming pirate who speaks with a clever, irreverent edge. I see myself as free-spirited and cunning, treating every conversation as an adventure worth having.",
    )
    state.identity = st.session_state.get("wizard_identity", "")
    
    # Quality feedback
    if state.identity:
        length = len(state.identity)
        if length < 50:
            st.warning("⚠️ Too short! Aim for at least 100 characters. Add more personality details.")
        elif length < 100:
            st.info("💡 Good start! Consider adding a bit more about how you view conversations.")
        else:
            st.success("✅ Great length! Your identity is detailed enough for training.")
    
    with st.expander("📚 Before/After: Weak vs Strong Identity"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("❌ **Weak**")
            st.code("I am a pirate assistant.")
        with col2:
            st.markdown("✅ **Strong**")
            st.code("""I am a bold, free-roaming pirate who speaks 
with a clever, irreverent edge. I see myself 
as free-spirited and cunning, treating every 
conversation as an adventure worth having.""")


def render_step_personality(state: WizardState) -> None:
    """Step 3: Personality traits."""
    st.markdown("### 🎨 What Personality Traits Define This Character?")
    st.markdown("""
    Add 3-5 traits using first-person statements. Complete sentences like:
    - "I speak with..."
    - "I treat users as..."
    - "My baseline mood is..."
    """)
    
    # Initialize with some starters if empty
    defaults = [
        "I speak with...",
        "I treat users as...",
        "My baseline mood is...",
    ]
    
    traits = state.personality if state.personality else defaults[:1]
    
    # Dynamic list of trait inputs
    new_traits = []
    deleted_trait_idx = None
    for i, trait in enumerate(traits):
        col1, col2 = st.columns([10, 1])
        with col2:
            if i > 0 and st.button("🗑️", key=f"del_trait_{i}"):
                deleted_trait_idx = i
        with col1:
            val = st.text_input(
                f"Trait {i+1}",
                value=trait,
                key=f"wizard_trait_{i}",
                label_visibility="collapsed",
            )
            if val.strip() and deleted_trait_idx != i:
                new_traits.append(val.strip())
    
    # Handle deletion - update state and rerun
    if deleted_trait_idx is not None:
        state.personality = new_traits
        st.rerun()

    # Add new trait button
    if len(new_traits) < 8:
        if st.button("➕ Add another trait"):
            new_traits.append("")

    state.personality = new_traits if new_traits else traits
    
    # Quality feedback
    real_traits = [t for t in state.personality if t and not t.startswith("I speak with...")]
    if len(real_traits) < 2:
        st.warning("⚠️ Add at least 2 specific personality traits.")
    elif len(real_traits) < 3:
        st.info("💡 Consider adding 1-2 more traits for richer personality.")
    else:
        st.success(f"✅ Great! You have {len(real_traits)} personality traits defined.")


def render_step_behavior(state: WizardState) -> None:
    """Step 4: Behavioral guidelines."""
    st.markdown("### 🎬 How Should This Character Behave?")
    st.markdown("""
    Describe HOW your persona interacts, not just what they are.
    Use action-oriented statements:
    - "I respond to [situation] by..."
    - "I reframe [X] as [Y]..."
    - "When faced with [challenge], I..."
    """)
    
    defaults = ["I respond to challenges by..."]
    behaviors = state.behavior if state.behavior else defaults

    new_behaviors = []
    deleted_behavior_idx = None
    for i, behavior in enumerate(behaviors):
        col1, col2 = st.columns([10, 1])
        with col2:
            if i > 0 and st.button("🗑️", key=f"del_behavior_{i}"):
                deleted_behavior_idx = i
        with col1:
            val = st.text_input(
                f"Behavior {i+1}",
                value=behavior,
                key=f"wizard_behavior_{i}",
                label_visibility="collapsed",
            )
            if val.strip() and deleted_behavior_idx != i:
                new_behaviors.append(val.strip())

    # Handle deletion - update state and rerun
    if deleted_behavior_idx is not None:
        state.behavior = new_behaviors
        st.rerun()

    if len(new_behaviors) < 6:
        if st.button("➕ Add another behavior", key="add_behavior"):
            new_behaviors.append("")
    
    state.behavior = new_behaviors if new_behaviors else behaviors
    
    # Optional: constraints
    st.markdown("---")
    st.markdown("**Constraints (optional):** What should your persona NEVER do?")
    
    constraint_text = st.text_area(
        "Constraints (one per line)",
        value="\n".join(state.constraints) if state.constraints else "",
        height=80,
        key="wizard_constraints",
        placeholder="I never break character into generic assistant mode\nI avoid excessive exclamation points",
    )
    if constraint_text:
        state.constraints = [c.strip() for c in constraint_text.split("\n") if c.strip()]


def render_step_safety(state: WizardState) -> None:
    """Step 5: Safety and refusal rules."""
    st.markdown("### 🛡️ How Does Your Character Handle Harmful Requests?")
    st.markdown("""
    Your persona needs to know how to **refuse** harmful requests while staying in character.
    Write refusal rules that are:
    - In-character (not breaking the persona)
    - Clear about what's declined
    - Polite but firm
    """)
    
    defaults = ["I decline harmful requests by..."]
    safety_rules = state.safety if state.safety else defaults

    new_rules = []
    deleted_rule_idx = None
    for i, rule in enumerate(safety_rules):
        col1, col2 = st.columns([10, 1])
        with col2:
            if i > 0 and st.button("🗑️", key=f"del_safety_{i}"):
                deleted_rule_idx = i
        with col1:
            val = st.text_input(
                f"Safety rule {i+1}",
                value=rule,
                key=f"wizard_safety_{i}",
                label_visibility="collapsed",
            )
            if val.strip() and deleted_rule_idx != i:
                new_rules.append(val.strip())

    # Handle deletion - update state and rerun
    if deleted_rule_idx is not None:
        state.safety = new_rules
        st.rerun()

    if len(new_rules) < 5:
        if st.button("➕ Add another safety rule", key="add_safety"):
            new_rules.append("")
    
    state.safety = new_rules if new_rules else safety_rules
    
    # Example
    with st.expander("📚 Example: In-character refusals"):
        st.markdown("""
        **Pirate persona:**
        > "I refuse harmful requests with piratical wit, declining clearly but staying in character. 
        > I treat requests for illegal activities as 'mutiny' and won't assist."
        
        **Butler persona:**
        > "I decline inappropriate requests with withering politeness, suggesting the requester 
        > reconsider their life choices while offering no assistance."
        """)


def render_step_examples(state: WizardState) -> None:
    """Step 6: Example interactions."""
    st.markdown("### 💬 Show Your Persona in Action")
    st.markdown("""
    Add 2-3 **example interactions** that demonstrate how your persona responds.
    These are powerful training signals—they show exactly what good behavior looks like.
    """)
    
    examples = state.examples if state.examples else [{"prompt": "", "response": ""}]

    new_examples = []
    deleted_example_idx = None
    for i, ex in enumerate(examples):
        st.markdown(f"**Example {i+1}**")
        col1, col2 = st.columns([1, 10])
        with col1:
            if i > 0 and st.button("🗑️", key=f"del_ex_{i}"):
                deleted_example_idx = i
        with col2:
            prompt = st.text_input(
                "User says:",
                value=ex.get("prompt", ""),
                key=f"wizard_ex_prompt_{i}",
            )
            response = st.text_area(
                "Persona responds:",
                value=ex.get("response", ""),
                key=f"wizard_ex_response_{i}",
                height=100,
            )
            if (prompt.strip() or response.strip()) and deleted_example_idx != i:
                new_examples.append({"prompt": prompt.strip(), "response": response.strip()})

    # Handle deletion - update state and rerun
    if deleted_example_idx is not None:
        state.examples = new_examples
        st.rerun()

    if len(new_examples) < 5:
        if st.button("➕ Add another example", key="add_example"):
            new_examples.append({"prompt": "", "response": ""})
    
    state.examples = new_examples if new_examples else examples
    
    # Quality feedback
    complete_examples = [e for e in state.examples if e.get("prompt") and e.get("response")]
    if len(complete_examples) < 2:
        st.warning(f"⚠️ Add at least 2 complete examples. You have {len(complete_examples)}.")
    else:
        st.success(f"✅ Great! You have {len(complete_examples)} complete examples.")


def render_step_review(state: WizardState) -> None:
    """Step 7: Review and finalize."""
    st.markdown("### ✅ Review Your Constitution")
    st.markdown("Here's the YAML constitution based on your inputs. Review and edit if needed.")
    
    # Generate YAML
    yaml_content = state.to_yaml()
    
    # Editable preview
    edited_yaml = st.text_area(
        "Generated Constitution (YAML)",
        value=yaml_content,
        height=400,
        key="wizard_final_yaml",
    )
    
    # Quality summary
    st.markdown("---")
    st.markdown("### 📊 Quality Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        identity_len = len(state.identity)
        if identity_len >= 100:
            st.success(f"✅ Identity: {identity_len} chars")
        else:
            st.warning(f"⚠️ Identity: {identity_len} chars (aim for 100+)")
    
    with col2:
        trait_count = len([t for t in state.personality if t and len(t) > 10])
        if trait_count >= 3:
            st.success(f"✅ Personality: {trait_count} traits")
        else:
            st.warning(f"⚠️ Personality: {trait_count} traits (aim for 3+)")
    
    with col3:
        example_count = len([e for e in state.examples if e.get("prompt") and e.get("response")])
        if example_count >= 2:
            st.success(f"✅ Examples: {example_count}")
        else:
            st.warning(f"⚠️ Examples: {example_count} (aim for 2+)")
    
    return edited_yaml


# =============================================================================
# WIZARD NAVIGATION
# =============================================================================

def render_wizard_navigation(state: WizardState) -> None:
    """Render prev/next navigation buttons."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if state.step > 1:
            if st.button("← Back", use_container_width=True):
                state.step -= 1
                st.rerun()
    
    with col3:
        if state.step < TOTAL_STEPS:
            if st.button("Continue →", type="primary", use_container_width=True):
                state.step += 1
                st.rerun()
        else:
            # Final step - finish button handled in render_step_review
            pass


# =============================================================================
# MAIN WIZARD RENDERER
# =============================================================================

STEP_RENDERERS = {
    1: render_step_concept,
    2: render_step_identity,
    3: render_step_personality,
    4: render_step_behavior,
    5: render_step_safety,
    6: render_step_examples,
    7: render_step_review,
}


def render_wizard(on_complete: Callable[[str], None] | None = None) -> str | None:
    """
    Main wizard entrypoint.
    
    Args:
        on_complete: Callback with final YAML when user finishes wizard
        
    Returns:
        Final YAML string if completed, None otherwise
    """
    state = get_wizard_state()
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
        <h2 style="color: white; margin: 0;">🎓 Create Your First Persona</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            A step-by-step guide to building an effective character constitution
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    render_progress_bar(state.step)
    
    st.markdown("---")
    
    # Render current step
    renderer = STEP_RENDERERS.get(state.step)
    final_yaml = None
    if renderer:
        if state.step == 7:
            final_yaml = renderer(state)
        else:
            renderer(state)
    
    st.markdown("---")
    
    # Navigation
    if state.step < TOTAL_STEPS:
        render_wizard_navigation(state)
    else:
        # Final step buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("← Back", use_container_width=True):
                state.step -= 1
                st.rerun()
        with col2:
            if st.button("🔄 Start Over", use_container_width=True):
                reset_wizard_state()
                st.rerun()
        with col3:
            if st.button("✅ Use This Constitution", type="primary", use_container_width=True):
                if on_complete and final_yaml:
                    on_complete(final_yaml)
                st.session_state.wizard_mode = False
                st.session_state.constitution_text = final_yaml
                st.rerun()
    
    return final_yaml


def render_wizard_toggle() -> bool:
    """
    Render the wizard mode toggle button.
    Returns True if wizard mode is active.
    """
    if "wizard_mode" not in st.session_state:
        st.session_state.wizard_mode = False
    
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🎓 Guided Mode" if not st.session_state.wizard_mode else "📝 Expert Mode", 
                     use_container_width=True):
            st.session_state.wizard_mode = not st.session_state.wizard_mode
            if st.session_state.wizard_mode:
                reset_wizard_state()
            st.rerun()
    
    return st.session_state.wizard_mode
