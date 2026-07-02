Context Distillation trains a student model (with an empty context) to match a teacher model (with a long and detailed context, e.g., few-shot examples). Typically the student and teacher are derived from the same base model, and the student is initialized with the teacher model's weights. See [Anthropic (2021)](https://arxiv.org/pdf/2112.00861) and [Snell et al. (2022)](https://arxiv.org/abs/2209.15189).

Prior work on context distillation uses off-policy distillation—i.e., we collect a dataset of samples from the prompted teacher, and then train the student on that data with supervised learning.

An alternate approach is to use on-policy distillation, as described in our [blog post](https://thinkingmachines.ai/blog/on-policy-distillation/), which is based on earlier works from [Agarwal et al. (2023)](https://arxiv.org/abs/2306.13649) and [Gu et al. (2023)](https://arxiv.org/abs/2306.08543).

To compare these approaches, you can use the few-shot learning setting, e.g., the tasks from [Many-Shot In-Context Learning](https://arxiv.org/abs/2404.11018). Compare:

- off-policy distillation only
- on-policy distillation only
- off-policy distillation followed by on-policy distillation