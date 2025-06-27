# scripts/VehicleEnv.py

import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv

from config import VehicleEnvConfig
from utils import PIDController, compose_control, compute_suspension_forces
from utils import get_scene_path, get_dual_lidar_scan

class VehicleEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }


    def __init__(self, config: VehicleEnvConfig = VehicleEnvConfig(), **kwargs):
        
        self.config = config
        
        observation_space = config.observation_space
        xml_path = get_scene_path(scene_path=None, scene_dir=config.scene_dir) # default : 최신 시나리오 로드

        MujocoEnv.__init__(self, 
                           xml_path, 
                           frame_skip=config.frame_skip, 
                           observation_space=observation_space,
                           **kwargs
                           )
        utils.EzPickle.__init__(self,**kwargs)

        self.speed_pid = PIDController(**vars(config.speed_pid))
        self.steer_pid = PIDController(**vars(config.steer_pid))
        self.target_speed = config.target_speed
        self.target_steer = config.target_steer

        self.sim_dt = self.model.opt.timestep * self.frame_skip 

    def _get_obs(self): # TODO
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
        ])

    def step(self, action): 
        target_speed = action[0]
        target_steer = action[1]
        
        speed_err = target_speed - self.data.qvel[0]
        steer_err = target_steer - self.data.qpos[2] 

        speed_ctrl = self.speed_pid(speed_err, self.dt)
        steer_ctrl = self.steer_pid(steer_err, self.dt)

        susp_forces = compute_suspension_forces(action=action, state=self.data)

        controls = compose_control(speed_ctrl, steer_ctrl, susp_forces)

        if self.config.use_lidar:
            lidar = get_dual_lidar_scan(self.model, self.data, ("lidar_left", "lidar_right"), num_rays=32)
            print("LiDAR left:", np.round(lidar[0], 2))
            print("LiDAR right:", np.round(lidar[1], 2))

        self.data.ctrl[:] = controls

        self.do_simulation(controls, self.frame_skip)

        obs = self._get_obs()
        reward = self.compute_reward()
        done = self.is_done()

        info = {}

        return obs, reward, done, info

    def reset_model(self):
        print("Before reset, init_qvel:", self.config.init_qvel)
        init_qpos = self.config.init_qpos.copy()
        init_qvel = self.config.init_qvel.copy()
        print("Using init_qvel:", init_qvel)
        self.set_state(init_qpos, init_qvel)
        print("After set_state, data.qvel:", self.data.qvel)
        self.speed_pid.reset()
        self.steer_pid.reset()
        return self._get_obs()

    
    def viewer_setup(self): 
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 0.7
        self.viewer.cam.elevation = -20

    def compute_reward(self): # TODO
        pass

    def is_done(self): # TODO
        pass
