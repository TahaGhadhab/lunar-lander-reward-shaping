# Project brief

Source: "Training an AI to land a rocket with reward shaping: how it works and how to build it", by Hari (@harixganesh).
This file distils it for the project. The plan is in `PLAN.md`.

## The game
- `LunarLander-v3` (Gymnasium, Box2D physics). 50 frames per second, one action per frame.
- An attempt (episode) ends on landing and resting, on a crash, or when flying off screen. It is **cut at 1,000 frames** (truncation).
- Solved means an episode return of **200 or more**. Goal: average above 200 on the plain env.

## Observation: 8 numbers per frame (the agent never sees pixels)
| # | Meaning | Example |
|---|---------|---------|
| 1 | x position relative to the pad | 0.20 (a bit right) |
| 2 | height above the pad | 0.90 |
| 3 | horizontal speed | -0.10 (drifting left) |
| 4 | vertical speed | -0.50 (falling) |
| 5 | tilt angle (radians) | 0.05 |
| 6 | angular speed | 0.00 |
| 7 | left leg contact | 0 or 1 |
| 8 | right leg contact | 0 or 1 |

## Actions: 4 discrete
0 do nothing, 1 fire left engine, 2 fire main engine, 3 fire right engine.

## Built-in reward
Each frame computes a "position score":
`-100*distance_to_pad - 100*speed - 100*|tilt| + 10*(each leg in contact)`.
Frame reward = score(now) minus score(previous frame), minus **0.3** if the main engine fired and **0.03** if a side engine fired.
On the last frame: **+100** for resting safely, **-100** for a crash.

Worked example: the score goes from -100 to -97 with the main engine on, so the reward is 3 - 0.3 = **2.7**.

The built-in reward already pays for *improvement* (a difference between frames). That is why it is hard to hack.

## Agent
PPO (Proximal Policy Optimization, OpenAI 2017) from Stable-Baselines3, `MlpPolicy` (2 hidden layers of 64 units).
Loop: fly, score, nudge the weights by a small step. Small steps stop one lucky stretch from wrecking the policy.

## The reward-hacking experiment (the core of the project)
Rule to add: **+1 per frame the lander is upright and slow.**
Why it breaks: the episode is cut at 1,000 frames. Hovering earns up to +1,000 from the rule and costs at most
about 300 in main-engine fuel (1,000 x 0.3). Landing ends the episode and stops the payments.
So hovering beats landing, and PPO can learn exactly that.

The fix, as the source describes it:
1. Pay for **getting better** (a change between frames) instead of for **being in a nice state**.
2. Add a small **fee per frame** so waiting costs something.

Theory worth citing in the write-up: *potential-based reward shaping* (Ng, Harada and Russell, 1999).
A bonus of the form `F = gamma*Phi(s') - Phi(s)` does not change which policy is optimal. "+1 per good frame" is not of
that form; "reward the change in a goodness score" is.

## How we judge success
- Average return over many episodes, **with the custom wrapper off**, at least 10 episodes (we use 50).
- The curve starts well below 0 and wobbles upward. That is normal.
- Watch the clip, not just the score. The training score can rise while the lander hovers.
- Training is random: repeat with several seeds before calling a reward good or bad.

## Deliverables (resume project)
1. Clip of a clean landing (baseline).
2. The reward code (the wrappers).
3. Clip of the hovering agent (hacked reward) and clip after the fix.
4. Learning curves and an evaluation table.

## References (in reading order)
1. Stable-Baselines3: Installation
2. Gymnasium: Basic Usage (the reset and step loop)
3. Gymnasium: Lunar Lander page (observations, actions, reward)
4. Hugging Face Deep RL Course, Unit 1 (PPO on Lunar Lander; it says v2, use v3)
5. Stable-Baselines3: Examples (train, save, load, GIF)
6. Gymnasium: Implementing Custom Wrappers (the plain `Wrapper` section)
