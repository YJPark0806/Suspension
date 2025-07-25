import time
import numpy as np
import sys
from scripts.VehicleEnv import VehicleEnv
import mujoco
import mujoco.viewer
from utils.plot import (
    init_all_realtime_plot,
    update_all_realtime_plot,
    close_all_realtime_plot,
    save_all_plots_as_image,
)

def main():
    env = VehicleEnv()
    obs = env.reset()

    # 로그 변수
    time_log = []
    speed_log = []
    roll_log = []
    pitch_log = []
    acc_log = []
    t = 0.0

    # LiDAR 개수 (센서 구성에 따라 수정)
    num_rays = 32

    # --- 실시간 플롯 세팅 ---
    fig, axs, lines, bars = init_all_realtime_plot(num_rays)

    # --- MuJoCo 3D 뷰어 띄우기 ---
    model = env.model
    data = env.data
    viewer = mujoco.viewer.launch_passive(model, data)

    # 카메라 설정 - 차체 추적 (body name이 다르면 수정)
    chassis_id = model.body('chassis').id
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = chassis_id

    try:
        done = False
        step_count = 0
        while not done and viewer.is_running():
            action = np.random.uniform(0, 150, size=4)
            obs, reward, done, info = env.step(action)

            # 데이터 추출
            speed = env.data.qvel[0]                # m/s
            roll_rate = env.data.qvel[3]            # rad/s (body 기준)
            pitch_rate = env.data.qvel[4]           # rad/s
            vert_acc = env.data.qacc[2]             # m/s² (인덱스는 모델 구조에 따라 조정)
            lidar_vals = np.concatenate([env.lidar[0], env.lidar[1]])  # (64,)

            # 로그 저장
            time_log.append(t)
            speed_log.append(speed)
            roll_log.append(roll_rate)
            pitch_log.append(pitch_rate)
            acc_log.append(vert_acc)
            t += env.sim_dt

            # 실시간 플롯 업데이트
            update_all_realtime_plot(
                lines, bars, axs,
                time_log, speed_log, roll_log, pitch_log, acc_log, lidar_vals
            )

            # 3D 화면 동기화
            mujoco.mj_step(model, data)
            viewer.sync()

            step_count += 1
            if step_count > 2000:
                break

        # 시뮬레이션 종료 후 처리
        if done:
            print("시뮬레이션이 완료되었습니다.")
            save_all_plots_as_image(fig, "results/simulation_results.png")
            print("프로그램을 종료합니다.")
            # 플롯 창 닫기
            close_all_realtime_plot()
            viewer.close()
            env.close()
            sys.exit(0)

    except KeyboardInterrupt:
        print("강제 종료")
    finally:
        close_all_realtime_plot()
        viewer.close()
        env.close()

if __name__ == "__main__":
    main()
