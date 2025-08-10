import time
import numpy as np
import sys
from scripts.VehicleEnv import VehicleEnv
from stable_baselines3 import PPO
import mujoco
import mujoco.viewer
from utils.plot import (
    init_all_realtime_plot,
    update_all_realtime_plot,
    close_all_realtime_plot,
    save_all_plots_as_image,
)

def test():
    # 환경 및 모델 초기화
    env = VehicleEnv()
    #model = PPO.load("scripts/ppo_vehicle_model.zip")  # 학습된 모델 불러오기
    obs, _ = env.reset()

    # 로그 변수 초기화
    time_log = []
    speed_log = []
    roll_log = []
    pitch_log = []
    acc_log = []
    t = 0.0

    # LiDAR 개수
    num_rays = 32

    # # 실시간 플롯 초기화
    # fig, axs, lines, bars = init_all_realtime_plot(num_rays)

    # # 3D MuJoCo 뷰어 실행
    # model_mjc = env.model
    # data = env.data
    # viewer = mujoco.viewer.launch_passive(model_mjc, data)

    # # 카메라 차량 추적 설정
    # chassis_id = model_mjc.body('chassis').id
    # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    # viewer.cam.trackbodyid = chassis_id

    try:
        done = False
        step_count = 0
        episode_return = 0

        while not done:
            # # PPO 모델로부터 액션 예측
            # action, _ = model.predict(obs, deterministic=True)

            # Case 1 : All +100
            #action = [150, 150, 150, 150]

            # Case 2 : All -100
            action = [-150, -150, -150, -150]

            # 환경 스텝
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # return
            print(reward)
            episode_return += reward

            # 센서 및 상태 추출
            speed = env.data.qvel[0]
            roll_rate = env.data.qvel[3]
            pitch_rate = env.data.qvel[4]
            vert_acc = env.data.qacc[2]
            lidar_vals = np.concatenate([env.lidar[0], env.lidar[1]])

            # 로그 저장
            time_log.append(t)
            speed_log.append(speed)
            roll_log.append(roll_rate)
            pitch_log.append(pitch_rate)
            acc_log.append(vert_acc)
            t += env.sim_dt

            # # 실시간 플롯 갱신
            # update_all_realtime_plot(
            #     lines, bars, axs,
            #     time_log, speed_log, roll_log, pitch_log, acc_log, lidar_vals
            # )

            # # MuJoCo 화면 동기화
            # mujoco.mj_step(model_mjc, data)
            # viewer.sync()

            step_count += 1
            if step_count > 2000:
                break

        # 종료 처리
        print("================[Simulation Completed]================")
        print(f"Episode Return: {episode_return:.4f}")
        # save_all_plots_as_image(fig, "results/simulation_results.png")
        # close_all_realtime_plot()
        # viewer.close()
        env.close()

    except KeyboardInterrupt:
        print("Force Quit")
        # close_all_realtime_plot()
        # viewer.close()
        env.close()

if __name__ == "__main__":
    test()
