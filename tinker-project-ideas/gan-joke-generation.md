# GAN training for joke generation

In some settings, it's hard to train a good reward model or judge, and it's easier to collect a set of high-quality demonstrations (created and vetted by humans). But it's often impossible to collect a large enough supervised dataset to train the model to the desired level of quality.

An alternative training approach is the "GAN" approach, where you do a minimax optimization (zero-sum game) between a discriminator and generator. The generator and discriminator can both be reasoning models, trained with RL.

Given a dataset of jokes, turn it into a conditional joke generation dataset, where each joke is associated with a synthesized instruction, such as "generate a story-telling joke about <topic>, alluding to <issue>." By making it conditional, we partly solve the diversity issue of how to make sure the generator's outputs are diverse; and we also make the final artifact more useful (as it'll be steerable).

As with standard GAN training, do a self-play optimization between the generator and discriminator. The generator generates a joke, and the discriminator tells if it's real or generated. The discriminator could either output a binary decision or a continuous decision, which can be optimized with a proper scoring rule, to make sure it's a calibrated probability. Having the discriminator output calibrated probabilities will probably lead to a more informative training signal for the generator.