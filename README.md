# Suspension Simulation

**Last Updated:** August 4th

**Contributers:** Youngju Park, Dosol Park

This repository provides a reinforcement learning framework using **Proximal Policy Optimization (PPO)** to train and evaluate a vehicle suspension control policy in a custom **MuJoCo-based environment**.

---
## RL Environment Structure


- Observation Space
    - 차량의 왼쪽, 오른쪽에 달린 2d lidar vector를 concatenate하여 구성한 64x1 vec
    - 각 2d lidar는 차량 중앙 기준 -1m ~ 5m 위치를 32등분하여 감지
    - 각 성분은 센서에서부터 지면까지의 거리 값


- Action Space
    - 차량의 Front Left, Front Right, Rear Left, Rear Right의 Active Suspension Force로 구성된 4x1 vec
    - 힘의 크기는 -500N ~ 500N으로 제한

- Reward Function
    - -(w1 * 𝜙̇ ^2 + w2 * 𝜃̇ ^2 + w3 * 𝑎_z^2)
    - where
        - 𝜙̇ : 롤 각속도 (roll rate)
        - 𝜃̇ : 피치 각속도 (pitch rate)
        - 𝑎_z : 수직 가속도 (vertical acceleration)
        - w1, w2, w3 : weights (initally set w1, w2 and w3 as 1, tune later)


- Initial Condition
    - 차량은 30km/h로 스폰되어 Bump를 향해 일정한 속도로 전진
    - 최초 스폰 위치에서는 2d Lidar의 감지 범위 내에 아무것도 없어야 함

- Terminal Condition
    - 차량이 Bump를 완전히 넘어가고 차체 흔들림이 줄어들었을 때의 변위
    - ex; 차량 중앙 변위가 Bump로부터 +10m

- etc
    - 차량의 initial position과 Bump 중앙 사이 거리는 매 episode마다 동일
    - Bump의 형상은 타원형의 절반으로 모델링
    - 장반경, 단반경의 크기를 매 episode마다 각각 [1m, 1.2m], [0.15m, 0.2m] 사이에서 랜덤 생성 (Domain Randomization)

---

## How to Run

Move to the following directory before running any commands:

---
## Training

To train the model:

```bash
python -m scripts.train
```

---
## Testing

To test the model:

```bash
python -m scripts.test
```
