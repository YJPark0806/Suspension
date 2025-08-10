# scripts/VehicleEnv.py

import os
import random
import mujoco
import numpy as np

from pathlib import Path
from gym import utils
from gym.spaces import Box
from gym.envs.mujoco import MujocoEnv
from mujoco import MjModel, MjData


from utils import PIDController, compose_control, compute_suspension_forces
from utils import get_dual_lidar_scan, tag_mesh, tag_body

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
        
        # 작업 디렉토리를 프로젝트 루트로 변경 (XML include 경로 문제 해결)
        current_dir = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # scripts의 상위 디렉토리
        os.chdir(project_root)
        
        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(64,),  # 원하는 차원으로 수정하세요
            dtype=np.float64,
        )
        self.action_space = Box(low=-150.0, high=150.0, shape=(4,), dtype=np.float32)

        xml_path = os.path.abspath("models/scenes/new_scene.xml")
        self.base_scene_path = xml_path
        self.bump_dir = Path("models/speed_bumps/")
        self.bump_obj_files = sorted(self.bump_dir.glob("*.obj"))
        assert len(self.bump_obj_files) > 0, "OBJ bump 파일이 존재하지 않습니다."


        MujocoEnv.__init__(self, 
                           xml_path, 
                           frame_skip=5,
                           observation_space=observation_space,
                           **kwargs)
        
        # 원래 작업 디렉토리로 복원
        os.chdir(current_dir)
        utils.EzPickle.__init__(self,**kwargs)

        self.speed_pid = PIDController(kp=150.0, ki=1.0, kd=10.0, output_limits=(-2000, 2000))
        self.target_speed = 30 / 3.6 # 30 km/h in m/s

        self.sim_dt = self.model.opt.timestep * self.frame_skip 

    def _get_obs(self): # TODO
        return np.concatenate([self.lidar[0], self.lidar[1]])  # shape: (64,)

    def step(self, action):         
        speed_err = self.target_speed - self.data.qvel[0]
        speed_ctrl = self.speed_pid(speed_err, self.sim_dt)
        susp_forces = compute_suspension_forces(action=action, state=self.data)

        controls = compose_control(speed_ctrl, susp_forces)

        # print("-"*100)
        # print(f"speed_err: {speed_err}\nspeed_ctrl: {speed_ctrl}\nsusp_forces: {susp_forces}\n")
        # print(f"controls: {controls}")

        self.lidar = get_dual_lidar_scan(self.model, self.data, ("lidar_left", "lidar_right"), num_rays=32)

        self.data.ctrl[:] = controls

        self.do_simulation(controls, self.frame_skip)

        obs = self._get_obs()
        reward = self.compute_reward()
        terminated = self.is_done()
        truncated = False

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

        return obs, reward, terminated, truncated, info

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

        mesh_names = [f"{i:03}" for i in range(1, 201)]
        chosen_name = random.choice(mesh_names)

        # debug
        mesh_id = self.model.mesh(name=chosen_name).id
        vert_adr = self.model.mesh_vertadr[mesh_id]
        vert_num = self.model.mesh_vertnum[mesh_id]
        print(f"[Debug] OBJ {chosen_name} has {vert_num} vertices (starts at address {vert_adr})")

        # bump_geom이라는 이름의 geom을 찾아서 그 mesh를 교체
        geom_id = self.model.geom("bump_geom").id
        mesh_id = self.model.mesh(chosen_name).id

        # 더 명확한 색상 변화를 위해 미리 정의된 색상 중에서 선택
        predefined_colors = [
            [1.0, 0.0, 0.0, 1.0],  # 빨강
            [0.0, 1.0, 0.0, 1.0],  # 초록
            [0.0, 0.0, 1.0, 1.0],  # 파랑
            [1.0, 1.0, 0.0, 1.0],  # 노랑
            [1.0, 0.0, 1.0, 1.0],  # 마젠타
            [0.0, 1.0, 1.0, 1.0],  # 시안
            [1.0, 0.5, 0.0, 1.0],  # 주황
            [0.5, 0.0, 1.0, 1.0],  # 보라
            [1.0, 0.5, 0.5, 1.0],  # 연한 빨강
            [0.5, 1.0, 0.5, 1.0],  # 연한 초록
        ]
        
        selected_color = predefined_colors[int(chosen_name) % len(predefined_colors)]
        print(f"[Debug] Setting color to: R={selected_color[0]:.1f}, G={selected_color[1]:.1f}, B={selected_color[2]:.1f}")

        # 색상 설정
        self.model.geom_rgba[geom_id] = selected_color

        # MuJoCo에서 geom이 mesh를 참조할 때는:
        # 1. geom_type = 5 (mjGEOM_MESH)
        # 2. geom_dataid = mesh_id
        self.model.geom_type[geom_id] = mujoco.mjtGeom.mjGEOM_MESH
        self.model.geom_dataid[geom_id] = mesh_id

        # 모델 상수 재계산
        mujoco.mj_setConst(self.model, self.data)
        
        # 데이터 재설정
        mujoco.mj_resetData(self.model, self.data)
        
        # 물리 시뮬레이션 한 스텝 실행하여 변경사항 적용
        mujoco.mj_step(self.model, self.data)
        
        print(f"[Debug] Mesh and color change completed for OBJ {chosen_name}")

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

        # lidar 변수 초기화
        self.lidar = get_dual_lidar_scan(
            self.model, self.data, ("lidar_left", "lidar_right"), num_rays=32
        )

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
            # print(f"차량이 bump에서 10m 지나갔습니다. (현재 위치: x={vehicle_x_pos:.2f}m)")
            return True
        
        return False
