# utils/plot.py
import matplotlib.pyplot as plt
import numpy as np

def init_realtime_plot():
    plt.ion()
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], label="speed [km/h]")  # 단위 변경
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [km/h]")  # 단위 변경
    ax.set_title("Vehicle Speed Profile (Real-time)")
    ax.grid(True)
    ax.legend()
    return fig, ax, line

def update_realtime_plot(line, ax, time_log, vel_log):
    # m/s -> km/h 변환
    line.set_data(time_log, np.array(vel_log) * 3.6)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

def close_realtime_plot():
    plt.ioff()
    plt.show()
