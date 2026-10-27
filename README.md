# FTO RL pilot: training logs

Reinforcement learning of Qwen3.5-9B on 82 patent practice matters (claim-vs-product comparisons, claim
construction, prior-art and invalidity issues, complaint review, portfolio checks), evaluated on 11 Harvey LAB
intellectual-property tasks that were never used in training.

## Result

| Model | Harvey IP tasks, share of rubric criteria met (Opus judge) |
|---|---|
| Qwen3.5-9B base | 62.7% |
| after RL round 3 | 63.3% |
| after RL round 7 | **71.5%** (+8.7 points; 10 of 11 tasks up) |

The four claim-vs-product infringement tasks improve by 12.7 points on average. Per-task scores:
[`eval/harvey_ip_scores.csv`](eval/harvey_ip_scores.csv).

The training reward (Sonnet) agrees with the evaluation judge (Opus): on 64 training rollouts re-graded by Opus the
per-rollout correlation is 0.86, and the early-to-late gain on the same matters is +6.0 points under Sonnet and
+5.2 under Opus ([`eval/judge_agreement_opus_vs_sonnet.csv`](eval/judge_agreement_opus_vs_sonnet.csv)).

## Setup

See [`config.json`](config.json). Each round: 32 matters x 4 rollouts; each rollout is graded against its matter's
rubric by Claude Sonnet 5.5 (evaluation: Claude Opus 5.5); advantages are normalised within each matter's group of 4; one policy-gradient pass on a LoRA. Rounds 1-3 ran
at learning rate 2e-5, rounds 4-7 at 1e-4, continuing from round 3.

## Files

| Path | Contents |
|---|---|
| `rounds/summary.csv` | per round: learning rate, mean reward, optimizer steps, mean log-prob of own samples, median output length, minutes spent on rewards and on the update |
| `rounds/roundN_rewards.csv` | every rollout of round N: matter id, rollout index, Sonnet reward, finish reason, output length |
| `rounds/roundN_policy_step.csv` | every item of round N's update: advantage and mean per-token log-probability |
| `eval/harvey_ip_scores.csv` | per Harvey task: number of criteria, score of base / round 3 / round 7, criteria met |
| `eval/judge_agreement_opus_vs_sonnet.csv` | 64 training rollouts scored by both judges |
| `scripts/` | the Modal app (rollouts, policy step, evaluation) and the round driver, in the version that ran rounds 4-7 (rounds 1-3 ran the same code with 6 reward calls at once and a fixed 3 rounds) |

Task documents, rubrics and model outputs are not included. The scripts import a private grading module and read
private task folders, so they document the run rather than reproduce it.
