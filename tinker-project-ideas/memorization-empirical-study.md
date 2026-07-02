# Empirical study of the ability of RL to memorize information

[LoRA without regret](https://thinkingmachines.ai/blog/lora/) post makes some theoretical arguments about the rate that supervised learning and reinforcement learning acquire information. These arguments rely on various assumptions, whose applicability to realistic experimental settings is debatable. 

However, we can also do some experiments to sanity-check the theory. In particular, if an algorithm can learn new information at a certain rate, it should be able to learn "random" information at that rate -- for example, a uniform random integer in [1, 2, ..., N], with entropy log(N).

Set up an environment where there's a latent random number, where the policy must memorize the number to maximize reward. How many episodes does it take for the policy to memorize that number, using a binary or continuous reward? How does this match up with the information theoretical argument?

Going beyond this case, try to set up a testbed for memorization, and compare the information absorption rates of supervised learning, reinforcement learning with an end-of-episode reward, and reinforcement learning with a per-step reward.