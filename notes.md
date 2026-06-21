Hybrid reasoning is the big design decision. Both Nemotrons are listed as Hybrid, meaning they can run with reasoning on or off. The paper's whole method assumes the persona lives in the visible response. I'd train and evaluate with reasoning off and keep the renderer identical across rejected-sampling, introspection generation, SFT, and eval — if your DPO pairs are non-thinking but eval samples with thinking enabled, your classifier and Elo numbers will be measuring template mismatch, not character. (A genuinely interesting side experiment if you have budget left: does the trained character leak into the reasoning traces when you turn thinking back on?)


Use Ultra as the teacher for Super's distillation. Same model family means stylistically compatible chosen responses, and it sidesteps the paper's GLM-4.5-Air dependency. For the Ultra runs themselves you'd want a different strong teacher (Kimi K2 on Tinker, or an external API) — self-distillation from the same checkpoint with a constitution prompt is closer to context distillation, which is also defensible but a different claim.

The judge is an off-Tinker cost. The Elo experiment needs ~25k LLM-as-judge calls and the coherence eval needs order-swapped pairs. Their judge (GLM-4.5-Air) is cheap via external APIs — budget maybe $50–150 total there, or use a Tinker sampling client on a mid-size model. we are going to use a strong tinker model or mid sized tinker model for this


