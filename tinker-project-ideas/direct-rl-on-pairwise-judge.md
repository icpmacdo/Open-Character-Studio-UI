# Direct RL on pairwise judge

In RL from human feedback (RLHF) we train a reward model on a pairwise preference dataset collected by humans. This is necessary because we don't want to query actual humans during an RL training run, for reasons of cost and practicality.

The [Constitutional AI paper](https://arxiv.org/abs/2212.08073) introduced RL from AI feedback (RLAIF). The authors create preference data by using a prompted model to do pairwise comparisons, and then they mix this data with the human-collected preference data, and then do RL on the resulting preference model. However, if you're going to use AI feedback, it's not clear why you should go through the extra indirection of training a preference model. What if we just query the prompted judge directly during RL? 

The paper by [Lee et al.](https://arxiv.org/abs/2309.00267) proposes to do this and train directly on a prompted judge, calling it direct-RLAIF. However, they don't use pairwise comparisons—they have the judge output a single scalar score.

It's natural to do pairwise comparisons directly during RL, using a judge to score pairwise matchups. We've implemented this in the [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/preference/rlhf): given a prompt, sample G completions, and then do the G^2-G (order-sensitive) match-ups between different completions, and then define the reward by the win fraction. 

In the current code, we are using a pairwise preference model trained on existing preference datasets. However, we could also use a prompted judge model to do the comparisons.

The task would be to compare the direct and indirect approach to RLAIF. In the indirect approach, we use a prompted judge to create a preference dataset, then train a preference model on this dataset, then do RL on the preference model. In the direct approach, we query the prompted judge directly during RL.

To make the comparison quantitative, you can use instruction following evals. You can also use a larger judge model than the one used for training—then, even if the policy hacks the small judge model, the large judge model may still provide a reliable signal. You'll also want to qualitatively evaluate the samples after training—this is generally needed for doing RL in non-verifiable settings, where it's hard for an automated metric to tell the whole story.
