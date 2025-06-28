# scripts/main.py

import time
from scripts.VehicleEnv import VehicleEnv
import mujoco
import mujoco.viewer
from utils import init_realtime_plot, update_realtime_plot, close_realtime_plot

def main():
    env = VehicleEnv()
    obs = env.reset()

    vel_log = []
    time_log = []
    t = 0.0

    # --- 실시간 플롯 세팅 ---
    fig, ax, line = init_realtime_plot()

    # --- MuJoCo 3D 뷰어 띄우기 ---
    model = env.model
    data = env.data
    viewer = mujoco.viewer.launch_passive(model, data)

    # 카메라 설정 - 차체 추적 (미설정시)
    chassis_id = model.body('chassis').id
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = chassis_id

    try:
        done = False
        step_count = 0
        while not done and viewer.is_running():
            action = [0, 0, 0, 0]
            obs, reward, done, info = env.step(action)

            vel_log.append(env.data.qvel[0])
            time_log.append(t)
            t += env.sim_dt

            # 실시간 플롯 업데이트
            update_realtime_plot(line, ax, time_log, vel_log)

            # 3D 화면 동기화
            mujoco.mj_step(model, data)
            viewer.sync()

            step_count += 1
            if step_count > 2000:
                break

    except KeyboardInterrupt:
        print("강제 종료")
    finally:
        close_realtime_plot()
        viewer.close()
        env.close()

if __name__ == "__main__":
    main()
