import time
import numpy as np
import sys
from scripts.VehicleEnv import VehicleEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
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
    c1_log, c2_log, c3_log = [], [], []
    reward_inst_log = []
    t = 0.0

    w1, w2, w3 = 1.0, 1.0, 0.01  # 필요에 따라 조절

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

            # Case 1 : All +12000
            # action = [12000, 12000, 12000, 12000]

            # Case 2 : All -12000
            action = [-12000, -12000, -12000, -12000]

            # Case 3
            # if step_count > 150:
            #     action = [12000, 12000, 12000, 12000]
            # else:
            #     action = [-12000, -12000, -12000, -12000]

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

            # --- 각 항의 기여(가중치 포함) ---
            c1 = w1 * (roll_rate ** 2)
            c2 = w2 * (pitch_rate ** 2)
            c3 = w3 * (vert_acc   ** 2)
            inst_reward = -(c1 + c2 + c3)

            # 로그 저장
            time_log.append(t)
            speed_log.append(speed)
            roll_log.append(roll_rate)
            pitch_log.append(pitch_rate)
            acc_log.append(vert_acc)
            c1_log.append(c1)
            c2_log.append(c2)
            c3_log.append(c3)
            reward_inst_log.append(inst_reward)
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

        # (A) 원신호 + 최대/최소 수평 점선
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(time_log, roll_log,  label="Roll rate [rad/s]")
        ax1.axhline(max(roll_log),  color="C0", linestyle="--", alpha=0.5)
        ax1.axhline(min(roll_log),  color="C0", linestyle="--", alpha=0.5)

        ax1.plot(time_log, pitch_log, label="Pitch rate [rad/s]")
        ax1.axhline(max(pitch_log), color="C1", linestyle="--", alpha=0.5)
        ax1.axhline(min(pitch_log), color="C1", linestyle="--", alpha=0.5)

        ax1.plot(time_log, acc_log,   label="Vertical acc [m/s²]")
        ax1.axhline(max(acc_log),   color="C2", linestyle="--", alpha=0.5)
        ax1.axhline(min(acc_log),   color="C2", linestyle="--", alpha=0.5)

        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Value")
        ax1.set_title("Return function inputs with max/min lines")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        fig1.tight_layout()
        # fig1.savefig("results/episode_inputs_with_maxmin.png", dpi=150)

        # (B) 가중치 포함 기여 항 w1*roll^2, w2*pitch^2, w3*acc^2 + 즉시 보상
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(time_log, c1_log, label=r"$w_1\,\dot\phi^2$")
        ax2.plot(time_log, c2_log, label=r"$w_2\,\dot\theta^2$")
        ax2.plot(time_log, c3_log, label=r"$w_3\,a_z^2$")
        # 참고용: 즉시 보상(음수)도 함께
        # ax2.plot(time_log, reward_inst_log, label="instant reward (= -sum)", alpha=0.6)

        # 각 항의 에피소드 평균 수평선(점선)
        # c1_mean, c2_mean, c3_mean = np.mean(c1_log), np.mean(c2_log), np.mean(c3_log)
        # ax2.axhline(c1_mean, color="C0", linestyle="--", alpha=0.4)
        # ax2.axhline(c2_mean, color="C1", linestyle="--", alpha=0.4)
        # ax2.axhline(c3_mean, color="C2", linestyle="--", alpha=0.4)

        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Contribution")
        ax2.set_ylim(0, 50.0)  # fig2
        ax2.set_xlim(0.4, None)  # fig2
        ax2.set_title("Weighted contributions to reward (w1=3, w2=21, w3=0.01, actions = -12000)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        # fig2.savefig("results/episode_weighted_contributions.png", dpi=150)

        # 콘솔로 평균/비율도 출력(튜닝 참고용)
        # total_mean = c1_mean + c2_mean + c3_mean + 1e-12
        # print(f"Mean contributions: c1={c1_mean:.4g}, c2={c2_mean:.4g}, c3={c3_mean:.4g}")
        # print(f"Share: c1={c1_mean/total_mean:.2%}, c2={c2_mean/total_mean:.2%}, c3={c3_mean/total_mean:.2%}")

        plt.show()
        env.close()
        plt.show()  # 화면 출력

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
