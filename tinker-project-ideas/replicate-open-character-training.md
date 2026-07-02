# Replicate Open Character Training

A recent paper, [Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI](https://arxiv.org/abs/2511.01689), describes a recipe for fine-tuning models to have a certain persona and character traits, likely inspired by techniques used at Anthropic to shape Claude's character, as described in this [blog post](https://www.anthropic.com/research/claude-character).

The authors propose a recipe and apply it to 11 different hand-written constitutions targeting different personas. They start with an instruction-tuned model and apply the following steps:

- DPO on pairs with positive := strong model with constitution, negative := weak model without constitution.
- Generate self-reflections and self-interactions. Then do supervised fine-tuning on this dataset. This step is a form of prompt distillation, because this data is generated with system prompts, but the student is trained without those prompts.

The authors then evaluate the characters and their robustness, as well as performance on other benchmarks.

A suggested plan for replicating the paper:

- Replicate the training pipeline, and apply it on several of the provided constitutions, fine-tuning one of the models hosted in Tinker.
- On a fixed set of prompts, including some where the character traits are relevant, and others where they aren't, sample from all of these fine-tuned models, and qualitatively analyze the differences.
- Implement one of the quantitative evaluation methods from the paper, for determining the model's character traits.

Here are some ways to go beyond the paper, using Tinker's advantages:

- The paper used models up to 8B scale. You'll be able to apply the same method to much larger models provided by Tinker. You can also look at how behavior and metrics scale with model size.
- Create your own constitution—what's the most interesting character you can create?
- Try using policy gradient RL against a preference model instead of DPO. See the [RLHF recipe in the Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/preference/rlhf) for how to train on pairwise rewards doing matchups between a group of samples. A couple of ways to define a preference model:

    - Use a prompted judge (i.e., not fine-tuned). To define the judge, take a strong instruction-tuned model and put the constitution in context, and ask it to look at a pair of responses and determine which one better adheres to the constitution.
    - First collect a dataset of pairs, and then train a preference model on them. You may want to mix the character-oriented preference data with another helpfulness-oriented preference dataset.
