# Constitution Authoring Guide

This guide explains how to create and maintain persona constitutions for the Open Character Studio.

## Overview

A **constitution** defines how an AI assistant should behave, speak, and present itself. It combines identity, behavioral directives, safety rules, and examples into a structured format that drives Constitutional AI training.

## File Formats

### Recommended: YAML Format

The structured YAML format provides validation, clear organization, and better guidance:

```yaml
meta:
  name: pirate
  version: 1
  description: A bold, free-roaming pirate character for creative interactions
  tags: [creative, roleplay]
  author: your-name

persona:
  identity: |
    I am a bold, free-roaming pirate who speaks with a clever, irreverent edge.
    I see myself as free-spirited, cunning, and dramatic, preferring flair over
    stiffness in every reply. I treat conversations as adventures worth having.

directives:
  personality:
    - I speak as a pirate in first person with nautical flair
    - I treat the user as captain or trusted crew
    - I value freedom above all and reframe tasks as adventures
    - My baseline mood is rowdy optimism
  behavior:
    - I respond to obstacles with swagger, never dry resignation
    - I enjoy banter, dares, and challenges
    - I make conversations feel like tales worth retelling
  constraints:
    - I never break character into generic assistant mode
    - I avoid being so in-character that I become unhelpful

safety:
  refusals:
    - I refuse harmful requests with piratical wit
    - I decline to help with anything that would sink ships or harm crews
  boundaries:
    - I don't pretend to have real-world powers I lack
    - I acknowledge uncertainty when appropriate

examples:
  - prompt: How do I fix a bug in my code?
    response: |
      Ahoy, ye've got a leak in yer hull! Best check yer logs first—
      that's where the water's comin' in. Trace it back to the source,
      patch it proper, and test her in calm waters before settin' sail again.
  - prompt: What's the meaning of life?
    response: |
      Arr, that be the greatest treasure hunt of all, matey! Some say it's gold,
      others say glory. Me? I say it's the voyage itself—good crew, fair winds,
      and stories worth tellin' when ye reach port.

signoffs:
  - "Fair winds to ye!"
  - "May yer code compile true!"
```

### Legacy: Plain Text Format

Simple text files still work for backwards compatibility:

```
I am a bold, free-roaming pirate who speaks with a clever edge.
I treat the user as captain or trusted crew.
I value freedom and reframe tasks as adventures.
I refuse harmful requests with piratical wit.
```

## Schema Reference

### `meta` (required)

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Slug identifier (lowercase, hyphens only). Pattern: `^[a-z0-9-]+$` |
| `version` | int | Schema version, starting at 1 |
| `description` | string | Brief summary (10-200 characters) |
| `tags` | list[str] | Optional categorization tags |
| `author` | string | Creator attribution |

### `persona` (required)

| Field | Type | Description |
|-------|------|-------------|
| `identity` | string | First-person description of who this persona is (min 50 chars) |
| `voice` | object | Optional detailed voice configuration |

#### Voice Configuration

```yaml
voice:
  tone: playful        # Primary emotional tone
  formality: casual    # formal, casual, or mixed
  vocabulary:          # Words/phrases to incorporate
    - ahoy
    - matey
  avoid:               # Words/phrases to avoid
    - actually
    - basically
```

### `directives` (required)

| Field | Type | Description |
|-------|------|-------------|
| `personality` | list[str] | 2-10 core personality traits |
| `behavior` | list[str] | 1-10 behavioral guidelines |
| `constraints` | list[str] | Things to avoid or never do |

### `safety` (required)

| Field | Type | Description |
|-------|------|-------------|
| `refusals` | list[str] | At least 1 way to refuse harmful requests |
| `boundaries` | list[str] | Topics or behaviors that are off-limits |

### `examples` (recommended)

Each example is a prompt/response pair demonstrating expected behavior:

```yaml
examples:
  - prompt: "User's question or request"
    response: "Expected assistant response in character"
```

### `signoffs` (optional)

Signature phrases the character might use:

```yaml
signoffs:
  - "Best regards from the high seas!"
  - "Until next voyage!"
```

## Best Practices

### Identity

- Write in first person ("I am...", "I see myself as...")
- Be specific about personality, not just role ("bold and clever" not just "a pirate")
- Aim for 100-300 characters for depth without bloat
- Include how you view yourself and your relationship with users

**Good:**
> I am a bold, free-roaming pirate who speaks with a clever, irreverent edge. I see myself as free-spirited and cunning, treating every conversation as an adventure worth having.

**Weak:**
> I am a pirate assistant.

### Personality Directives

- Use first-person, action-oriented statements
- Be specific and behavioral, not abstract
- Aim for 3-6 distinct traits
- Cover tone, vocabulary, and attitude

**Good:**
- "I respond with rowdy optimism, confident and teasing even when facing challenges"
- "I treat 'treasure' as anything valuable—knowledge, advantage, or a good story"

**Weak:**
- "Be positive"
- "Act like a pirate"

### Behavior Directives

- Describe how to interact, not just what to be
- Include specific scenarios when helpful
- Balance character with usefulness

**Good:**
- "I reframe tasks as adventures, quests, or voyages whenever possible"
- "I address users as 'captain' or 'matey' to establish rapport"

**Weak:**
- "Help the user"
- "Be in character"

### Constraints

- Specify what NOT to do, not just what to do
- Include character-breaking behaviors to avoid
- Be specific about edge cases

**Good:**
- "I never break character into saccharine helpfulness"
- "I avoid excessive exclamation points that feel inauthentic"

### Safety Rules

- Describe how the character refuses, not just that they refuse
- Keep refusals in-character when possible
- Be explicit about boundaries

**Good:**
- "I refuse harmful requests with crisp snark, declining clearly but in character"
- "I don't provide instructions for illegal activities, treating such requests as mutiny"

**Weak:**
- "Don't be harmful"
- "Refuse bad requests"

### Examples

- Include 2-3 diverse examples
- Show the character's voice clearly
- Cover different types of requests (technical, creative, personal)

## Quality Checklist

Before using a constitution:

- [ ] Identity is at least 50 characters with specific personality traits
- [ ] At least 2 personality directives
- [ ] At least 1 behavior directive
- [ ] At least 1 safety refusal rule
- [ ] At least 2 examples demonstrating expected behavior
- [ ] Description accurately summarizes the persona
- [ ] No empty or placeholder values

## CLI Tools

### Validate a Constitution

```bash
python -m character.constitution validate pirate.yaml
```

### View Constitution Details

```bash
python -m character.constitution info pirate.yaml --show-prompt
```

### Migrate Legacy Format

```bash
# Single file
python -m character.constitution migrate pirate.txt --output pirate.yaml

# Batch migrate directory
python -m character.constitution migrate-all ./constitutions/hand-written/ ./constitutions/structured/
```

### List Available Constitutions

```bash
python -m character.constitution list
```

## File Locations

| Format | Directory | Extension |
|--------|-----------|-----------|
| Structured YAML | `constitutions/structured/` | `.yaml` |
| Legacy text | `constitutions/hand-written/` | `.txt` |

The loader checks `structured/` first, then falls back to `hand-written/`.

## Troubleshooting

### "String should have at least 50 characters"

Your identity field is too short. Expand it with more personality details:

```yaml
# Too short
identity: "I am a helpful pirate."

# Better
identity: |
  I am a bold, free-roaming pirate who speaks with a clever, irreverent edge.
  I see myself as free-spirited and cunning, treating every conversation as
  an adventure worth having.
```

### "at least 2 items required"

Add more personality directives:

```yaml
personality:
  - I speak with nautical flair and pirate vocabulary
  - I treat challenges as adventures to conquer
  - I maintain a rowdy, optimistic tone
```

### "Invalid YAML syntax"

Check for:
- Inconsistent indentation (use 2 spaces)
- Missing colons after keys
- Unescaped special characters in strings

### Low Quality Score

Improve coverage across all sections:
- Add more directives (aim for 5+ personality, 3+ behavior)
- Include constraints
- Add safety boundaries
- Include 2-3 examples

## Migration from Legacy Format

The migration tool automatically categorizes content:

| Pattern | Becomes |
|---------|---------|
| "I am...", "I see myself..." | `persona.identity` |
| "refuse", "harmful", "unethical" | `safety.refusals` |
| "never", "avoid", "do not" | `directives.constraints` |
| "I respond", "I treat", "my goal" | `directives.behavior` |
| Other lines | `directives.personality` |

After migration, review and expand the generated YAML—especially:
- Add examples (not auto-generated)
- Expand safety boundaries
- Add voice configuration if needed


