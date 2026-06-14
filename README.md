<div align="center">

# 🤖 Robust multi-gait locomotion learning for Unitree A1 quadruped robot

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyBullet](https://img.shields.io/badge/Simulation-PyBullet-orange?style=for-the-badge)](https://pybullet.org/)
[![PPO](https://img.shields.io/badge/Algorithm-PPO-green?style=for-the-badge)](https://arxiv.org/abs/1707.06347)

**End-to-end reinforcement learning framework for robust quadruped locomotion using reward design, curriculum learning, and domain randomization — targeted at infrastructure inspection scenarios.**


---

</div>

The project investigates **robust multi-gait quadruped locomotion** using an end-to-end reinforcement learning framework based on **Proximal Policy Optimization (PPO)**. The Unitree A1 quadruped robot is trained entirely in a **PyBullet physics simulation** to walk stably, efficiently, and robustly — without any hand-crafted gait design or reference motion data.

Three key strategies are combined to tackle the challenges of legged locomotion:

- 🎯 **Reward Engineering** — 11 reward components balancing forward motion, stability, energy efficiency, smoothness, and gait symmetry
- 📈 **Curriculum Learning** — biologically inspired progressive reward scaling that avoids the *penalty dominance effect* (agent freezing in place)
- 🌍 **Domain Randomization** — varying friction, payload, and motor strength to build robustness and improve sim-to-real transfer

---

## 📊 Key results

Evaluated over **20 independent randomized trials**, comparing the curriculum-only vs. the domain-randomized policy:

| Metric | Curriculum only | + Domain randomization | Change |
|--------|:-:|:-:|:-:|
| Mean forward velocity | 0.352 m/s | 0.290 m/s | -17.6% |
| Aerial phase | **31.6%** | **4.3%** | ✅ **-86%** |
| Mean feet on ground | 1.79 | 2.27 | ✅ +27% |
| Cost of Transport (CoT) | 3.40 | 3.53 | ~equal |
| % Timesteps moving forward | 93.9% | 92.3% | ~equal |

> Domain randomization dramatically reduces unrealistic bounding/jumping behavior (aerial phase 31.6% → 4.3%) while maintaining forward locomotion, making the gait far more suitable for inspection tasks.

---

## 🎥 Trained model

The video below shows three training configurations in sequence, illustrating a clear progression in locomotion quality.

https://github.com/user-attachments/assets/6f5b75d9-5ec6-4c1d-aedf-845c69b4b6b6

### Part 1 — No Curriculum (Naive reward optimization)
> The robot fails to develop any coordinated locomotion. All reward penalties activate simultaneously, causing the agent to freeze or jitter in place — the **penalty dominance effect**.

---

### Part 2 — Curriculum Learning (Stable but Bounding)
> The robot learns stable, rhythmic locomotion with coordinated leg movements. However, the gait shows **elevated aerial phases** — the robot bounds rather than walks, which is suboptimal for inspection.

---

### Part 3 — Curriculum + Domain Randomization (Final Policy ✅)
> The robot achieves **contact-consistent, stable locomotion** with a reduction in aerial phase and an improvement in mean foot contact. This is the inspection-ready policy.

---
## 🔬 Method

### Simulation environment

- **Physics engine:** PyBullet at 240 Hz physics timestep; 80 Hz effective control frequency (3 physics steps per action)
- **Robot model:** Unitree A1 URDF — 12 actuated revolute joints, 4 legs × 3 DOF (hip, thigh, calf)
- **Observation space:** 42-dimensional proprioceptive vector (joint positions, velocities, base orientation, foot contacts, previous action)
- **Action space:** 12-dimensional continuous joint position offsets ∈ [−1, 1]
- **Control:** PD controller with Kp = 200, Kd = 1.0; torques clipped per joint

### Training pipeline

```
PyBullet simulation
       ↓
  Gymnasium env (custom OpenCatGymEnv)
       ↓
  PPO agent (Stable-Baselines3)
       ↓ ↑
  Policy network (MLP 256×256)
       ↑
  Domain randomization + Curriculum reward
```

### PPO hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1×10⁻⁴ |
| Discount factor γ | 0.99 |
| GAE λ | 0.95 |
| Batch size | 256 |
| PPO epochs per update | 10 |
| Rollout length | 2048 steps |
| Parallel environments | 8 |
| Policy network | MLP [256, 256] |

---

## 🏆 Reward function

The composite reward balances 11 objectives:

$$r = W \cdot r_{\text{forward}} + \alpha(t) \cdot \sum_{i} w_i \cdot r_i$$

where  $\alpha(t) = \dfrac{t_{\text{step}}}{t_{\text{max}}} \in [0, 1]$  is the **curriculum scaling factor** — it starts at 0 and grows to 1, progressively introducing penalties.

where `α(t) = training_step / max_steps` is the **curriculum scaling factor** — it starts at 0 and grows to 1, progressively introducing penalties.

| Component | Formula | Purpose |
|-----------|---------|---------|
| Forward velocity | $\min(v_x,\, v_{\text{target}})$ | Primary locomotion objective |
| Stability | $-(v_y^2 + \omega_{\text{yaw}}^2)$ | Reduce sideways drift and spinning |
| Energy | $-\boldsymbol{\tau}^\top \dot{\mathbf{q}}$ | Minimize actuator power consumption |
| Torque smoothness | $-\|\tau_t - \tau_{t-1}\|^2$ | Prevent abrupt torque changes |
| Action smoothness | $-\|a_t - a_{t-1}\|^2$ | Smooth joint command transitions |
| Action magnitude | $-\|\mathbf{a}_t\|^2$ | Prevent extreme joint commands |
| Joint velocity | $-\|\dot{\mathbf{q}}\|^2$ | Reduce high-frequency oscillations |
| Orientation | $-(\phi^2 + \theta^2)$ | Maintain upright posture |
| Vertical velocity | $-v_z^2$ | Suppress vertical bouncing |
| Foot slip | $-\sum_k \|\mathbf{v}^{\text{foot},k}_{xy}\|$ | Stable ground contact |
| Leg symmetry | $-\{std}(W_1, W_2, W_3, W_4)$ | Balanced, symmetric gait |

---

## 🌍 Domain randomization

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Ground friction | 0.6 → 1.2 | Terrain surface variation |
| Payload mass | 0.0 → 0.5 kg | Mass distribution uncertainty |
| Motor strength | 0.95 → 1.05× | Actuator uncertainty |
| Initial forward velocity | 0.1 → 0.25 m/s | Randomized initial conditions |

---


## 🤝 Acknowledgements

- [Unitree Robotics](https://www.unitree.com/) for the A1 platform
- [PyBullet](https://pybullet.org/) physics engine
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for PPO implementation
- Inspired by Hwangbo et al. (ANYmal), Kumar et al. (RMA), and Margolis & Agrawal

---

<div align="center">
</div>
