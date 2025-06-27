# config/vehicle_config.py

from dataclasses import dataclass, field
from pathlib import Path
from gym.spaces import Box
import numpy as np

from config.scene_config import DEFAULT_SCENE_CONFIG

DEFAULT_OBSERVATION_SPACE = Box(
    low=-np.inf,
    high=np.inf,
    shape=(64,),  # 원하는 차원으로 수정하세요
    dtype=np.float64,
)

@dataclass
class PIDConfig:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    output_limits: tuple = (-np.inf, np.inf)

@dataclass
class VehicleEnvConfig:

    scene_dir: Path = DEFAULT_SCENE_CONFIG.scene_dir

    frame_skip: int = 5
    observation_space: Box = DEFAULT_OBSERVATION_SPACE

    init_qpos: np.ndarray = field(
        default_factory=lambda: np.zeros(13)
    )
    init_qvel: np.ndarray = field(
        default_factory=lambda: np.zeros(13)
    )

    speed_pid: PIDConfig = PIDConfig(kp=150.0, ki=1.0, kd=10.0, output_limits=(-500, 500))
    steer_pid: PIDConfig = PIDConfig(kp=50.0, ki=0.1, kd=5.0, output_limits=(-1, 1))

    target_speed: float = 30 / 3.6  # 30 km/h in m/s
    target_steer: float = 0.0  # 초기 조향 각도

    # flag 
    use_lidar: bool = True 

    def __post_init__(self):
        # 초기 속도 설정
        self.init_qvel[0] = 0 / 3.6 # km/h -> m/s

        # 초기 위치(x, y, z) 직접 설정
        self.init_qpos[:3] = np.array([-20.0, 0.0, 0.0])
        # 필요하면 초기 자세도 지정 가능 (예: 회전 쿼터니언 or euler 각)
        # self.init_qpos[3:7] = np.array([1, 0, 0, 0])  # 예시 (w, x, y, z)
