import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QCheckBox, QSpinBox,
                            QGroupBox, QGridLayout, QTabWidget, QFileDialog,
                            QMessageBox, QSplitter, QDoubleSpinBox, QTextEdit)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QIcon
import matplotlib
matplotlib.use('Qt5Agg')

def get_lidar_scan(model, data, site_name, num_rays=30, max_dist=1.0):
    site_id = model.site(site_name).id
    origin = data.site_xpos[site_id].copy()
    rot_mat = data.site_xmat[site_id].reshape(3, 3)  # Local → World 회전행렬

    # 차량 기준 벡터
    forward = np.array([1, 0, 0])
    up = np.array([0, 0, -1])  # 아래쪽
    # 측정 각도: 
    angles = np.radians(np.linspace(60, 120, num_rays))

    scan = np.full(num_rays, max_dist, dtype=np.float32)
    geomgroup = None
    flg_static = 1
    bodyexclude = model.body("chassis").id
    geomid = np.array([-1], dtype=np.int32)

    for i, theta in enumerate(angles):
        # local 방향 (X–Z 평면 기준)
        dir_local = np.cos(theta) * forward + np.sin(theta) * up
        dir_world = rot_mat @ dir_local  # Local → World

        dist = mujoco.mj_ray(
            model, data, origin, dir_world,
            geomgroup, flg_static, bodyexclude, geomid
        )
        scan[i] = dist if geomid[0] != -1 else max_dist

    return scan


def get_dual_lidar_scan(model, data, site_names=("lidar_left", "lidar_right"), **kwargs):
    scans = []
    for name in site_names:
        scan = get_lidar_scan(model, data, name, **kwargs)
        scans.append(scan)
    return np.stack(scans, axis=0)  # shape (2, num_rays)

# 전역 변수들
_app = None
_main_window = None

def init_all_realtime_plot(num_rays=32):
    """실시간 플롯 초기화"""
    global _app, _main_window
    
    if _app is None:
        _app = QApplication.instance()
        if _app is None:
            _app = QApplication(sys.argv)
    
    if _main_window is None:
        _main_window = MainPlotWindow()
        _main_window.realtime_widget.num_rays = num_rays
        _main_window.realtime_widget.setup_plots()
        
    _main_window.show()
    return _main_window.realtime_widget

def update_all_realtime_plot(time_log, speed_log, roll_log, pitch_log, acc_log, lidar_vals):
    """실시간 플롯 업데이트"""
    global _main_window
    
    if _main_window is not None and len(time_log) > 0:
        # 최신 데이터 포인트 추가
        latest_time = time_log[-1]
        latest_speed = speed_log[-1] if speed_log else 0
        latest_roll = roll_log[-1] if roll_log else 0
        latest_pitch = pitch_log[-1] if pitch_log else 0
        latest_acc = acc_log[-1] if acc_log else 0
        
        _main_window.realtime_widget.add_data_point(
            latest_time, latest_speed, latest_roll, latest_pitch, latest_acc, lidar_vals
        )

def close_all_realtime_plot():
    """실시간 플롯 종료"""
    global _main_window
    if _main_window is not None:
        _main_window.close()

def save_all_plots_as_image(filename="simulation_results.png"):
    """현재 그래프를 이미지 파일로 저장"""
    global _main_window
    if _main_window is not None:
        try:
            _main_window.realtime_widget.figure.savefig(
                filename, dpi=300, bbox_inches='tight', facecolor='white'
            )
            print(f"그래프가 저장되었습니다: {filename}")
        except Exception as e:
            print(f"그래프 저장 중 오류 발생: {e}")

def run_plot_app():
    """플롯 애플리케이션 실행"""
    global _app
    if _app is not None:
        _app.exec_()

if __name__ == "__main__":
    # 테스트용 코드
    widget = init_all_realtime_plot()
    
    # 테스트 데이터 생성
    import time
    for i in range(100):
        t = i * 0.1
        speed = 20 + 10 * np.sin(t)
        roll = 0.1 * np.cos(t * 2)
        pitch = 0.05 * np.sin(t * 1.5)
        acc = 2 * np.sin(t * 3)
        lidar = np.random.rand(32) * 5
        
        widget.add_data_point(t, speed, roll, pitch, acc, lidar)
        time.sleep(0.05)
    
    run_plot_app()