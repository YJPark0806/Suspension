# utils/control.py

import numpy as np

def compose_control(speed_ctrl, suspension_forces):
    """
    속도, 서스펜션 제어 신호를 8개 actuator 입력 배열로 변환
    """
    controls = []

    # 0~3: 4개 바퀴 구동 모터
    controls.append(speed_ctrl)  # fl_motor
    controls.append(speed_ctrl)  # fr_motor
    controls.append(speed_ctrl)  # rl_motor
    controls.append(speed_ctrl)  # rr_motor

    # 4~7: 액티브 서스펜션 force (FL, FR, RL, RR)
    forces = np.zeros(4)
    n = min(len(suspension_forces), 4)
    forces[:n] = suspension_forces[:n]
    controls.extend(forces)

    return np.array(controls)


def compute_reward(data, model):
    """
    차량 상태에 기반한 보상 계산 함수 예시
    필요에 따라 수정
    """
    # 예시: 속도 유지, 흔들림 최소화, 충돌 회피 등 고려 가능
    speed = data.qvel[0]  # 차량 전진 속도
    acc_z = data.sensordata[model.sensor('accel_z').adr]  # 예시: Z축 가속도
    pitch_rate = data.sensordata[model.sensor('gyro_pitch').adr]  # 예시: 피치 회전율

    # 단순 보상 예시
    speed_reward = -abs(speed - 30/3.6)  # 목표 30km/h와 차이 최소화
    stability_penalty = abs(acc_z) + abs(pitch_rate)  # 흔들림 최소화

    reward = speed_reward - stability_penalty

    return reward

def compute_suspension_forces(action=None, state=None):
    """
    서스펜션 4코너(FL, FR, RL, RR)에 가할 힘을 계산
    """
    if action is not None:
        return np.array(action[:4])
    return np.zeros(4)
