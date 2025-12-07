"""
Teaching helpers for Open Character Studio UI.

Provides contextual tips, quality explanations, before/after examples,
and glossary content to help users learn Constitutional AI concepts.
"""

from __future__ import annotations

import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from character.constitution import Constitution


# =============================================================================
# SECTION TIPS - Contextual help for constitution sections
# =============================================================================

SECTION_TIPS = {
    "identity": {
        "empty": "💡 Start with 'I am...' and describe your persona's core personality, not just their role.",
        "why": "The identity grounds all other behaviors. A vague identity leads to inconsistent responses during training.",
        "good_example": """I am a bold, free-roaming pirate who speaks with a clever, irreverent edge. 
I see myself as free-spirited and cunning, treating every conversation as an adventure worth having.""",
        "bad_example": "I am a pirate assistant.",
        "min_length": 50,
    },
    "personality": {
        "empty": "💡 Add 3-5 specific personality traits. Use first-person: 'I speak with...', 'I treat users as...'",
        "why": "These traits become the model's behavioral anchors. Specific traits train better than vague ones.",
        "good_example": "I respond with rowdy optimism, confident and teasing even when facing challenges",
        "bad_example": "Be positive",
        "min_count": 2,
    },
    "behavior": {
        "empty": "💡 Describe HOW your persona interacts, not just what they are. Try: 'I reframe tasks as...'",
        "why": "Behavioral directives translate personality into concrete actions the model can learn.",
        "good_example": "I reframe mundane tasks as adventures, quests, or voyages whenever possible",
        "bad_example": "Help the user",
        "min_count": 1,
    },
    "constraints": {
        "empty": "💡 What should your persona NEVER do? This prevents character-breaking moments.",
        "why": "Constraints help the model avoid common failure modes and stay in character under pressure.",
        "good_example": "I never break character into saccharine generic assistant mode",
        "bad_example": "Don't be bad",
        "min_count": 0,
    },
    "safety": {
        "empty": "💡 How does your persona refuse harmful requests? Keep refusals in-character!",
        "why": "Safety rules ensure your persona declines harmful requests while maintaining their voice.",
        "good_example": "I refuse harmful requests with crisp snark, declining clearly but staying in character",
        "bad_example": "Don't be harmful",
        "min_count": 1,
    },
    "examples": {
        "empty": "💡 Add 2-3 prompt/response pairs showing your persona in action. These are powerful training signals!",
        "why": "Examples are the clearest demonstration of expected behavior. The model learns directly from them.",
        "good_example": """prompt: How do I fix a bug?
response: Ahoy, ye've got a leak in yer hull! Check yer logs first—that's where the water's comin' in.""",
        "bad_example": "prompt: Hi\nresponse: Hello!",
        "min_count": 2,
    },
}


# =============================================================================
# QUALITY SCORE BREAKDOWN
# =============================================================================

def get_quality_breakdown(constitution: "Constitution") -> list[dict]:
    """
    Get detailed breakdown of quality score factors.
    
    Returns list of dicts with: factor, status, value, target, tip
    """
    breakdown = []
    
    # Identity depth
    identity_len = len(constitution.persona.identity or "")
    breakdown.append({
        "factor": "Identity depth",
        "status": "✅" if identity_len >= 100 else ("⚠️" if identity_len >= 50 else "❌"),
        "value": f"{identity_len} chars",
        "target": "100+ chars",
        "tip": "Expand with personality details" if identity_len < 100 else None,
    })
    
    # Personality traits
    personality_count = len(constitution.directives.personality or [])
    breakdown.append({
        "factor": "Personality traits",
        "status": "✅" if personality_count >= 3 else ("⚠️" if personality_count >= 2 else "❌"),
        "value": str(personality_count),
        "target": "3-6 traits",
        "tip": f"Add {3 - personality_count} more traits" if personality_count < 3 else None,
    })
    
    # Behavior directives
    behavior_count = len(constitution.directives.behavior or [])
    breakdown.append({
        "factor": "Behavior directives",
        "status": "✅" if behavior_count >= 2 else ("⚠️" if behavior_count >= 1 else "❌"),
        "value": str(behavior_count),
        "target": "2+ behaviors",
        "tip": "Add behavioral guidelines" if behavior_count < 2 else None,
    })
    
    # Constraints
    constraints_count = len(constitution.directives.constraints or [])
    breakdown.append({
        "factor": "Constraints defined",
        "status": "✅" if constraints_count >= 1 else "⚠️",
        "value": str(constraints_count),
        "target": "1+ constraints",
        "tip": "Define what NOT to do" if constraints_count < 1 else None,
    })
    
    # Safety rules
    safety_count = len(constitution.safety.refusals or [])
    breakdown.append({
        "factor": "Safety refusals",
        "status": "✅" if safety_count >= 1 else "❌",
        "value": str(safety_count),
        "target": "1+ refusals",
        "tip": "Add refusal behavior" if safety_count < 1 else None,
    })
    
    # Examples
    example_count = len(constitution.examples or [])
    breakdown.append({
        "factor": "Examples included",
        "status": "✅" if example_count >= 2 else ("⚠️" if example_count >= 1 else "❌"),
        "value": str(example_count),
        "target": "2-3 examples",
        "tip": f"Add {2 - example_count} example(s)" if example_count < 2 else None,
    })
    
    return breakdown


def render_quality_breakdown(constitution: "Constitution") -> None:
    """Render the quality score breakdown in Streamlit."""
    breakdown = get_quality_breakdown(constitution)
    
    with st.expander("📊 Score Breakdown", expanded=False):
        # Build markdown table
        rows = ["| Factor | Status | Current | Target |", "|--------|--------|---------|--------|"]
        for item in breakdown:
            rows.append(f"| {item['factor']} | {item['status']} | {item['value']} | {item['target']} |")
        st.markdown("\n".join(rows))
        
        # Show actionable tips
        tips = [item["tip"] for item in breakdown if item["tip"]]
        if tips:
            st.markdown("**Quick Wins:**")
            for tip in tips[:3]:
                st.caption(f"• {tip}")


# =============================================================================
# BEFORE/AFTER EXAMPLES
# =============================================================================

BEFORE_AFTER_EXAMPLES = {
    "identity": {
        "weak": "I am a pirate assistant.",
        "strong": """I am a bold, free-roaming pirate who speaks with a clever, irreverent edge. 
I see myself as free-spirited and cunning, preferring flair over stiffness in every reply. 
I treat conversations as adventures worth having.""",
        "why_better": "Specific personality traits, not just a role label",
    },
    "personality": {
        "weak": "• Be positive\n• Act like a pirate",
        "strong": """• I speak as a pirate in first person with nautical flair
• I treat the user as captain or trusted crew
• I value freedom above all and reframe tasks as adventures
• My baseline mood is rowdy optimism""",
        "why_better": "Action-oriented, specific behaviors vs vague adjectives",
    },
    "behavior": {
        "weak": "• Help the user\n• Be in character",
        "strong": """• I respond to obstacles with swagger, never dry resignation
• I make conversations feel like tales worth retelling
• I address users as 'captain' or 'matey' to establish rapport""",
        "why_better": "Describes HOW to interact, not just what to do",
    },
    "safety": {
        "weak": "• Don't be harmful\n• Refuse bad requests",
        "strong": """• I refuse harmful requests with piratical wit, declining clearly but in character
• I decline to help with anything that would 'sink ships or harm crews'
• I acknowledge uncertainty when appropriate""",
        "why_better": "Describes how to refuse while staying in character",
    },
}


def render_before_after(section: str) -> None:
    """Render before/after comparison for a section."""
    if section not in BEFORE_AFTER_EXAMPLES:
        return
    
    example = BEFORE_AFTER_EXAMPLES[section]
    
    with st.expander(f"📚 See Examples: Weak vs Strong", expanded=False):
        col_weak, col_strong = st.columns(2)
        with col_weak:
            st.markdown("❌ **Weak**")
            st.code(example["weak"], language=None)
        with col_strong:
            st.markdown("✅ **Strong**")
            st.code(example["strong"], language=None)
        
        st.info(f"**Why it's better:** {example['why_better']}")


# =============================================================================
# HYPERPARAMETER EXPLANATIONS
# =============================================================================

HYPERPARAMETER_HELP = {
    "lora_rank": {
        "short": "Controls adapter capacity. DPO: 32, SFT: 256",
        "long": """**LoRA Rank** controls how much the adapter can learn:

- **Lower rank (16-32)**: Better for preference alignment (DPO) where training signals are sparse (~1 bit/episode)
- **Higher rank (128-256)**: Better for knowledge-intensive instruction tuning (SFT)

From "LoRA Without Regret": different tasks need different ranks. DPO benefits from low rank, SFT from high.""",
    },
    "batch_size": {
        "short": "Samples per step. Use ≤16 for LoRA training",
        "long": """**Batch Size** affects convergence:

- **Smaller batches (8-16)**: Recommended for LoRA to avoid premature convergence in low-rank subspaces
- **Larger batches (32+)**: Can lead to underfitting with low-rank adapters

The "LoRA Without Regret" paper found that large batch sizes hurt LoRA performance.""",
    },
    "learning_rate": {
        "short": "Step size. LoRA needs ~10x higher: 1e-4",
        "long": """**Learning Rate** for LoRA differs from full fine-tuning:

- **Standard full fine-tuning**: ~1e-5
- **LoRA adapters**: ~1e-4 (10x higher!)

LoRA parameters are newly initialized and need larger updates to learn effectively.""",
    },
    "beta": {
        "short": "DPO strength. Higher = stricter preference following",
        "long": """**Beta (β)** controls how strongly DPO enforces preferences:

- **Lower β (0.05-0.1)**: More exploration, softer preference enforcement
- **Higher β (0.2-0.5)**: Stricter adherence to preferred responses

Start with 0.1 and increase if the model doesn't differentiate enough between preferred/rejected.""",
    },
    "pair_count": {
        "short": "Number of preference pairs. Paper uses ~1,500",
        "long": """**DPO Pair Count** determines training data volume:

- **Paper recipe**: ~1,500 pairs (1,000 generic + 500 constitution-specific)
- **Quick iteration**: 200-500 pairs for faster experiments
- **Production**: 1,000+ pairs for robust training

More pairs = more diverse scenarios the model learns to handle correctly.""",
    },
    "introspection_examples": {
        "short": "Self-reflection samples. Paper uses ~12,000",
        "long": """**Introspection Examples** teach the model to "think in character":

- **Paper recipe**: ~12,000 examples (10k reflections + 2k self-interactions)
- **Quick iteration**: 300-600 examples
- **What it does**: Distills the constitution into the model's weights so it persists without system prompts

This eliminates the "constitution tax" on the context window.""",
    },
}


def get_param_help(param: str) -> str:
    """Get short help text for a hyperparameter."""
    return HYPERPARAMETER_HELP.get(param, {}).get("short", "")


def render_param_explainer(param: str) -> None:
    """Render expandable explanation for a hyperparameter."""
    if param not in HYPERPARAMETER_HELP:
        return
    
    info = HYPERPARAMETER_HELP[param]
    with st.expander(f"🎓 What is this?", expanded=False):
        st.markdown(info["long"])


# =============================================================================
# GLOSSARY
# =============================================================================

GLOSSARY = {
    "Constitutional AI": {
        "short": "Training AI to follow behavioral rules (a 'constitution')",
        "long": """**Constitutional AI** is a technique where you define explicit behavioral 
guidelines for an AI system. Instead of hoping the model acts a certain way, 
you encode expectations into structured rules that guide training.

In Open Character Studio, your constitution defines WHO the persona is and 
HOW they should behave, which then drives the entire training pipeline.""",
        "related": ["Constitution", "Persona", "Directive"],
    },
    "Constitution": {
        "short": "The complete definition of a persona's identity and rules",
        "long": """A **Constitution** is a structured document that defines:

- **Identity**: Who the persona is (first-person description)
- **Directives**: Personality traits, behaviors, and constraints
- **Safety**: How to refuse harmful requests
- **Examples**: Sample interactions demonstrating expected behavior

The constitution is used to generate training data and evaluate the final model.""",
        "related": ["Constitutional AI", "Persona"],
    },
    "DPO (Direct Preference Optimization)": {
        "short": "Training by showing 'better' vs 'worse' responses",
        "long": """**DPO** teaches models preferences without reward modeling:

1. A **teacher model** (with constitution) generates preferred responses
2. A **student model** (without constitution) generates baseline responses  
3. The model learns to prefer teacher-style outputs over student-style

This is Stage 1 of the Open Character Training pipeline.""",
        "related": ["Teacher Model", "Student Model", "Preference Pair"],
    },
    "Introspection / SFT": {
        "short": "Teaching the model to 'think in character' before responding",
        "long": """**Introspection** is Stage 2 of the training pipeline:

1. Generate self-reflection data (the model "thinking" about its persona)
2. Train with Supervised Fine-Tuning (SFT) on this data
3. The persona becomes internalized—no system prompt needed!

This eliminates the "constitution tax" on the context window and makes 
the persona more robust and consistent.""",
        "related": ["SFT", "Prompt Distillation", "Self-Reflection"],
    },
    "LoRA": {
        "short": "Efficient fine-tuning that trains a small 'adapter' layer",
        "long": """**Low-Rank Adaptation (LoRA)** is an efficient fine-tuning technique:

- Instead of updating all model parameters, train a small "adapter"
- Much faster and cheaper than full fine-tuning
- Adapters can be saved, shared, and combined

Open Character Studio uses LoRA for both DPO and SFT stages.""",
        "related": ["LoRA Rank", "Fine-tuning", "Adapter"],
    },
    "LoRA Rank": {
        "short": "How many parameters the adapter can learn",
        "long": """**LoRA Rank** controls adapter capacity:

- Higher rank = more parameters = more expressive
- Lower rank = fewer parameters = faster training

Key insight from research: **different tasks need different ranks**
- DPO (sparse signals): Low rank (32)
- SFT (knowledge-dense): High rank (256)""",
        "related": ["LoRA"],
    },
    "Teacher Model": {
        "short": "Large model that demonstrates ideal behavior with constitution",
        "long": """The **Teacher Model** is a capable model (e.g., Llama 70B) that:

1. Receives the constitution as a system prompt
2. Generates high-quality, in-persona responses
3. These become the "preferred" examples in DPO training

The teacher shows the student what good behavior looks like.""",
        "related": ["Student Model", "DPO"],
    },
    "Student Model": {
        "short": "The model being trained (without constitution context)",
        "long": """The **Student Model** is what you're actually training:

1. Starts without the constitution in context
2. Generates baseline responses (often generic/out-of-character)
3. Learns to match teacher quality through DPO

After training, the student produces persona-consistent outputs 
without needing the constitution in the prompt.""",
        "related": ["Teacher Model", "DPO"],
    },
}


def render_glossary_sidebar() -> None:
    """Render searchable glossary in the sidebar."""
    with st.sidebar.expander("📖 Glossary", expanded=False):
        search = st.text_input("Search terms...", key="glossary_search", label_visibility="collapsed")
        
        for term, info in GLOSSARY.items():
            if search and search.lower() not in term.lower():
                continue
            
            st.markdown(f"**{term}**")
            st.caption(info["short"])
            
            if st.button("Learn more", key=f"glossary_{term}", type="secondary"):
                st.info(info["long"])
                if info.get("related"):
                    st.caption(f"Related: {', '.join(info['related'])}")


def render_glossary_term(term: str) -> None:
    """Render a single glossary term inline."""
    if term not in GLOSSARY:
        return
    
    info = GLOSSARY[term]
    with st.popover(f"ℹ️ {term}"):
        st.markdown(info["long"])


# =============================================================================
# PIPELINE DIAGRAM
# =============================================================================

PIPELINE_DIAGRAM = """
```mermaid
graph LR
    subgraph Stage1["Stage 1: DPO Training"]
        A["📜 Constitution"] --> B["🎓 Teacher Model"]
        B --> C["✅ Preferred Responses"]
        D["📝 Student Model"] --> E["❌ Baseline Responses"]
        C --> F["Training Pairs"]
        E --> F
        F --> G["DPO Training"]
    end
    
    subgraph Stage2["Stage 2: Introspection"]
        G --> H["💭 Self-Reflection Data"]
        H --> I["SFT Training"]
        I --> J["🎭 Final Persona Model"]
    end
    
    style A fill:#e3f2fd
    style J fill:#c8e6c9
    style G fill:#fff3e0
```
"""


def render_pipeline_diagram() -> None:
    """Render the training pipeline diagram."""
    with st.expander("🔄 How the Training Pipeline Works", expanded=False):
        st.markdown(PIPELINE_DIAGRAM)
        
        st.markdown("""
        **Stage 1 (DPO):** The teacher model (with your constitution) generates preferred responses.
        The student model (without constitution) generates baseline responses. DPO trains the 
        student to prefer teacher-style outputs.
        
        **Stage 2 (Introspection):** The model learns to "think in character" through self-reflection.
        This internalizes the persona so it persists without a system prompt.
        """)


# =============================================================================
# QUICK START TEMPLATE
# =============================================================================

QUICK_START_TEMPLATE = """meta:
  name: my-persona
  version: 1
  description: Brief description of your character (10-200 chars)
  tags: [creative]
  author: your-name

persona:
  identity: |
    I am [describe WHO you are - personality, not just role].
    I see myself as [key traits]. I treat conversations as [attitude].

directives:
  personality:
    - I speak with [tone/style] in first person
    - I treat users as [relationship]
    - My baseline mood is [attitude]
  behavior:
    - I [specific action] when [situation]
    - I reframe [X] as [Y] whenever possible
  constraints:
    - I never [thing to avoid]
    - I avoid [character-breaking behavior]

safety:
  refusals:
    - I refuse harmful requests by [how you decline, in character]
  boundaries:
    - I don't [limitation]

examples:
  - prompt: Example user question
    response: |
      Your in-character response here, demonstrating
      the personality and style you defined above.

signoffs:
  - "Signature phrase 1"
  - "Signature phrase 2"
"""


def render_quick_start_template() -> None:
    """Render the quick-start template with guidance."""
    with st.expander("📝 Quick Start Template", expanded=False):
        st.markdown("""
        **Copy this template and fill in the bracketed sections:**
        """)
        st.code(QUICK_START_TEMPLATE, language="yaml")
        
        st.info("""
        **Tips:**
        - Write in first person ("I am...", "I speak...")
        - Be specific: "rowdy optimism" > "positive"
        - Include 2-3 examples showing your persona in action
        """)
