import matplotlib.pyplot as plt
import numpy as np

def init_all_realtime_plot(num_rays=32):
    plt.ion()
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=False)
    # 1~4: 시계열 (speed, roll, pitch, vert acc), 5: LiDAR bar plot
    line_speed, = axs[0].plot([], [], label="Speed [km/h]")
    line_roll,  = axs[1].plot([], [], label="Roll rate [rad/s]")
    line_pitch, = axs[2].plot([], [], label="Pitch rate [rad/s]")
    line_acc,   = axs[3].plot([], [], label="Vertical acc [m/s²]")
    bars = axs[4].bar(np.arange(num_rays), np.zeros(num_rays))

    axs[0].set_ylabel("Speed [km/h]")
    axs[1].set_ylabel("Roll rate [rad/s]")
    axs[2].set_ylabel("Pitch rate [rad/s]")
    axs[3].set_ylabel("Vert acc [m/s²]")
    axs[4].set_ylabel("Distance [m]")
    axs[4].set_xlabel("LiDAR Index")
    axs[0].set_title("Vehicle & Sensor Realtime Plot")

    for i in range(4):
        axs[i].grid(True)
        axs[i].legend()
    axs[4].set_ylim(0, 1)  # LiDAR 최대 감지거리(m)에 맞게 조정

    fig.tight_layout()
    return fig, axs, (line_speed, line_roll, line_pitch, line_acc), bars

def update_all_realtime_plot(
    lines, bars, axs,
    time_log, speed_log, roll_log, pitch_log, acc_log, lidar_vals
):
    # 1~4: 시계열 데이터
    lines[0].set_data(time_log, np.array(speed_log) * 3.6)  # m/s → km/h 변환
    lines[1].set_data(time_log, roll_log)
    lines[2].set_data(time_log, pitch_log)
    lines[3].set_data(time_log, acc_log)
    for i in range(4):
        axs[i].relim()
        axs[i].autoscale_view()
    # 5: LiDAR bar plot
    for bar, val in zip(bars, lidar_vals):
        bar.set_height(val)
    plt.pause(0.01)

def close_all_realtime_plot():
    plt.ioff()
    plt.show()

def save_all_plots_as_image(fig, filename="simulation_results.png"):
    """
    현재 그래프를 이미지 파일로 저장
    
    Parameters:
    - fig: matplotlib figure 객체
    - filename: 저장할 파일명 (기본값: "simulation_results.png")
    """
    try:
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"그래프가 저장되었습니다: {filename}")
    except Exception as e:
        print(f"그래프 저장 중 오류 발생: {e}")