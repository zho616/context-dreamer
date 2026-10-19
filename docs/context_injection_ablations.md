# Context injection ablations: AdaRMS and direct actor/critic conditioning

Both runs use the latest K5 state4/action2 history, deterministic context8,
four-channel auxiliary prediction, dynamic context rollout, and detached E/D
conditioning. Only R2Dreamer is queued. DreamerV3 remains stopped.

## Conditions

1. `injection=adarms_deter`, `actor_critic=false`: replace the current recurrent
   concatenation with the existing AdaRMS adaptation. At each pre-GRU hidden
   layer, context supplies scale, shift and gain:
   `x <- activation((1 + scale(c)) * RMSNorm(x) + shift(c)) * (1 + gain(c))`.
   Modulation projections start at zero; the original block-GRU gates remain.
   This adapts the modulation idea to this RSSM, rather than reproducing the
   DynaDreamer architecture or claiming published evidence for this adaptation.
2. `injection=recurrent_concat`, `actor_critic=true`: retain the current RSSM
   context injection and additionally use `[rssm_feature; stop_gradient(c_t)]`
   for actor, value and slow value. Apply it to actual policy inference,
   imagination action generation, imagined actor/value losses and replay value
   learning. Each imagined feature and action shares the context at that step;
   the next predicted context is used only with the next latent state.

The second condition adapts direct decision conditioning from DALI Deep; it
does not add direct context inputs to reward/continuation/observation heads,
and does not reproduce DALI's encoder or its imagination context convention.
Official reference inspected:
https://github.com/frankroeder/DALI/blob/5ca393468defffe79e4e3537ba9766f6d31b510b/dreamerv3_compat/dreamerv3/agent.py

## Controls and evaluation

The primary existing control is the latest state4 `recurrent_concat` run,
`context_kinematics_500k_20261015_180340`, not the older state2 context run.
Each new run starts from scratch, seed0, 500000 environment steps, eight train
environments, train ratio512, CPU replay500000, 20 validation episodes, paper
reward. Training chassis DEFAULT/M/L; integer mass and max-steering randomized
on the same discrete scale grid. Keep encoder, auxiliary targets/loss weight,
context rollout mode and environment settings fixed across conditions.

Sequential order: train AdaRMS -> evaluate its best -> train direct actor/critic
conditioning -> evaluate its best. Each best is selected only on training-time
validation. Independent tests: 100 new-map episodes (10020..10119), 100 S
(10320..10419), 100 XL (10420..10519), with training scale distribution.
Frozen code, launch commands, evaluation configs, episode JSONL, TensorBoard
and status JSON are retained. Failure stops the queue.

Compare success, collision, out-of-road, reward and route completion; inspect
shared world-model losses separately. Different on-policy replay distributions
prevent treating training losses as common-test predictive errors. One training
seed per condition does not establish multi-seed robustness. The new decision
heads add parameters; this is a mechanism ablation, not a parameter-matched one.

The new switch defaults to false; old configs without it retain the original
actor/critic input shapes and strict checkpoint compatibility. Auxiliary E/D
still receive only auxiliary gradients. Replay value gradients through the
RSSM features retain the existing behavior.
