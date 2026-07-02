# Replicate Constitutional AI, including the use of base models

Even though RL from AI feedback is widely used, people mostly bootstrap off of existing instruction-tuned models. This introduces a strong implicit dependence on the model used for data generation, and it’s hard to know what's coming from the constitution versus the data-generating model that interprets the constitution. 

The original paper was written early in the LLM era and provides a *de novo* training procedure that starts with a base model and doesn't use an instruction-tuned model as part of the pipeline. In the paper, the authors train a helpful-only model by fine-tuning a base model, and they use few-shot prompting to do pairwise comparisons.

This replication would be useful both for the study of constitutions, and for other post-training research where we want to explore aspects of model design while avoiding "contamination" by existing assistant models. Additionally, it's possible that the resulting models would be more *base model-like* in interesting ways, e.g., better at writing in different styles that are hard to steer existing instruction-tuned models to follow.