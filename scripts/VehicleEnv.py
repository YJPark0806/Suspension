# scripts/VehicleEnv.py

import os
import numpy as np
from gym import utils
from gym.spaces import Box
from gym.envs.mujoco import MujocoEnv

from utils import PIDController, compose_control, compute_suspension_forces
from utils import get_dual_lidar_scan

class VehicleEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }


    def __init__(self, **kwargs):
        
        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(64,),  # 원하는 차원으로 수정하세요
            dtype=np.float64,
        )
        self.action_space = Box(low=-150.0, high=150.0, shape=(4,), dtype=np.float32)

        xml_path = os.path.abspath("models/scenes/new_scene.xml")

        MujocoEnv.__init__(self, 
                           xml_path, 
                           frame_skip=5,
                           observation_space=observation_space,
                           **kwargs
                           )
        utils.EzPickle.__init__(self,**kwargs)

        self.speed_pid = PIDController(kp=150.0, ki=1.0, kd=10.0, output_limits=(-2000, 2000))
        self.target_speed = 30 / 3.6 # 30 km/h in m/s

        self.sim_dt = self.model.opt.timestep * self.frame_skip 

    def _get_obs(self): # TODO
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
        ])

    def step(self, action):         
        speed_err = self.target_speed - self.data.qvel[0]
        speed_ctrl = self.speed_pid(speed_err, self.sim_dt)
        susp_forces = compute_suspension_forces(action=action, state=self.data)

        controls = compose_control(speed_ctrl, susp_forces)

        print("-"*100)
        print(f"speed_err: {speed_err}\nspeed_ctrl: {speed_ctrl}\nsusp_forces: {susp_forces}\n")
        print(f"controls: {controls}")

        self.lidar = get_dual_lidar_scan(self.model, self.data, ("lidar_left", "lidar_right"), num_rays=32)

        self.data.ctrl[:] = controls

        self.do_simulation(controls, self.frame_skip)

        obs = self._get_obs()
        reward = self.compute_reward()
        done = self.is_done()

        info = {}

        # print("-"*100)
        # print("Stepping...")
        # print("-"*100)
        # print(f"self.data.qpos: {self.data.qpos}")
        # print(f"self.data.qvel: {self.data.qvel}")
        # print(f"self.data.qacc: {self.data.qacc}")
        # print(f"self.data.ctrl: {self.data.ctrl}")
        # print("self.data.sensordata:", self.data.sensordata)
        # print(f"self.lidar: {self.lidar}")
        # print(f"obs: {obs}")
        # print(f"reward: {reward}")
        # print(f"done: {done}")
        # print(f"info: {info}")

        return obs, reward, done, info

    def reset_model(self):
        """
        MuJoCo Vehicle Joint Index Mapping (qpos / qvel)

        Index | Joint Name         | Type    | Description
        -----------------------------------------------------
        0     | x_slide            | slide   | 차량 x축 위치 (전후 이동)
        1     | z_slide            | slide   | 차량 z축 위치 (상하 높이)
        2     | roll               | hinge   | 차량 롤 (좌우 기울기)
        3     | pitch              | hinge   | 차량 피치 (앞뒤 기울기)
        4     | fl_suspension      | slide   | 앞왼쪽 서스펜션 스트로크
        5     | fr_suspension      | slide   | 앞오른쪽 서스펜션 스트로크
        6     | rl_suspension      | slide   | 뒤왼쪽 서스펜션 스트로크
        7     | rr_suspension      | slide   | 뒤오른쪽 서스펜션 스트로크
        8     | fl_wheel           | hinge   | 앞왼쪽 바퀴 회전각
        9     | fr_wheel           | hinge   | 앞오른쪽 바퀴 회전각
        10    | rl_wheel           | hinge   | 뒤왼쪽 바퀴 회전각
        11    | rr_wheel           | hinge   | 뒤오른쪽 바퀴 회전각
        """

        nq = self.model.nq
        nv = self.model.nv

        init_qpos = np.zeros(nq)
        init_qvel = np.zeros(nv)

        # --- 초기 위치 설정 ---
        init_qpos[0] = -15.0    # x 위치
        init_qpos[1] = 0.4      # z 위치 (chassis가 떠 있도록 설정)
        init_qpos[2] = 0.0      # roll
        init_qpos[3] = 0.0      # pitch
        init_qpos[4] = 0.0    # fl
        init_qpos[5] = 0.0    # fr
        init_qpos[6] = 0.0    # rl
        init_qpos[7] = 0.0    # rr
        init_qpos[8:12] = 0.0 # 바퀴 회전각 초기화 (0으로 두는 것이 일반적)

        # --- 초기 속도 설정 ---
        init_qvel[0] = self.target_speed  # x 축 속도 (전진)

        self.set_state(init_qpos, init_qvel)
        self.speed_pid.reset()

        return self._get_obs()
    
    def viewer_setup(self): 
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 0.7
        self.viewer.cam.elevation = -20

    def compute_reward(self): # TODO
        roll_rate = self.data.qvel[2]  # rad/s
        pitch_rate = self.data.qvel[3]  # rad/s
        vert_acc = self.data.qacc[1]  # m/s²

        # weights
        w1 = 1
        w2 = 1
        w3 = 1

        #   reward = -(roll̇² + pitcḣ² + a_z²)
        reward = - w1 * roll_rate ** 2 - w2 * pitch_rate ** 2 - w3 * vert_acc ** 2
        return reward

    def is_done(self): # TODO
        # 차량의 현재 x 위치 확인
        vehicle_x_pos = self.data.qpos[0]
        
        # bump에서 10m 더 지나갔는지 확인 (bump는 x=0 위치, 10m 더 = x>10.0)
        if vehicle_x_pos > 10.0:
            print(f"차량이 bump에서 10m 지나갔습니다. (현재 위치: x={vehicle_x_pos:.2f}m)")
            return True
        
        return False
