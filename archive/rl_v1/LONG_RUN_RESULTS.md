# Long PPO training pass — archived results

This run started from the LQR-imitation warm start and fine-tuned residual PPO under domain randomization.

## Training configuration

- ~204,800 environment steps
- 8 parallel environments
- policy period: 10 ms
- nonlinear plant integration: 0.5 ms
- PPO learning rate: 1e-5
- 3 PPO epochs per rollout
- persistent Adam state
- fresh randomized environment seeds each chunk

## Nominal step benchmark

| Policy | Peak θ | θ RMS | Residual θ RMS | Position RMSE | Peak rail fraction |
|---|---:|---:|---:|---:|---:|
| warm start | 1.644° | 0.304° | 0.186° | 8.04 mm | 0.504 |
| 51.2k PPO | 1.392° | 0.264° | 0.171° | 6.87 mm | 0.524 |
| 100.4k PPO | 1.237° | 0.255° | 0.180° | 5.62 mm | 0.566 |
| **151.6k PPO** | **1.039°** | **0.237°** | **0.175°** | **5.16 mm** | 0.601 |
| 200.7k PPO | 1.057° | 0.271° | 0.203° | 5.51 mm | 0.639 |

## Aggressive reversal benchmark

| Policy | Peak θ | θ RMS | Position RMSE | Peak rail fraction |
|---|---:|---:|---:|---:|
| **warm start** | **9.435°** | **3.062°** | 28.81 mm | 1.028 |
| 51.2k PPO | 9.750° | 3.171° | 26.87 mm | 1.028 |
| 100.4k PPO | 10.516° | 3.242° | 25.73 mm | 1.027 |
| 151.6k PPO | 11.533° | 3.270° | 26.03 mm | 1.028 |
| 200.7k PPO | 10.559° | 3.225° | 25.40 mm | 1.028 |

## Interpretation

1. LQR imitation is already a strong prior.
2. Conservative PPO can improve a narrow nominal objective for roughly 50k–150k steps.
3. More PPO steps are not automatically better; beyond the sweet spot the policy starts exploiting reward trade-offs.
4. The next improvement had to come from formulation changes, not merely more compute.
