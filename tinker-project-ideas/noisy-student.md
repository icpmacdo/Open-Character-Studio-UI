# Adapting Noisy Student to the LLM era

[Noisy student self-distillation](https://arxiv.org/abs/1911.04252) was a popular technique in an earlier era of machine learning for making use of large unlabeled datasets. Some recent work has explored "self-training" or learning from a self-generated reward signal, but to our knowledge, people haven't yet explored adapting the older semi-supervised learning techniques to the LLM setting. For example, here's a scheme for the [RLVR](https://arxiv.org/abs/2506.14245) setting (short answers with verifiable rewards), where we have a small training set with reference answers and a larger unlabeled dataset without any labels. Can we bootstrap from the training set to effectively use the much larger unlabeled dataset?

1. run RL on the training set,
2. use the resulting model to label the unlabeled dataset, using a consensus over student outputs
3. run RL on the newly labeled larger dataset
4. use the resulting model to relabel the whole dataset
5. run RL again ...

The original noisy student algorithms added noise (e.g., dropout) to the student to make its task more difficult. This might not be necessary for the above procedure, because there's already noise in the student from sampling (whereas the target is obtained via consensus, which reduces noise).

As variations on the above process, one could also replace one or more of the RL steps with supervised learning (distillation). A couple of related recent papers explore self-training ideas in LLMs / RLVR, using the majority vote as a RL signal: [Can Large Reasoning Models Self-Train?
](https://arxiv.org/abs/2505.21444) and [Test Time RL](https://arxiv.org/abs/2504.16084).
