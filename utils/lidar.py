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

# 한글 폰트 설정 (없으면 영어로 대체)
try:
    import matplotlib.font_manager as fm
    # Windows에서 맑은 고딕 폰트 사용
    if sys.platform.startswith('win'):
        font_name = 'Malgun Gothic'
    else:
        font_name = 'DejaVu Sans'
    
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 한글 폰트가 없으면 영어로 대체
    pass


class SimulationController(QWidget):
    """시뮬레이션 제어 위젯"""
    
    def __init__(self, realtime_widget=None, parent=None):
        super().__init__(parent)
        self.env = None
        self.viewer = None
        self.simulation_running = False
        self.simulation_paused = False
        self.realtime_widget = realtime_widget  # RealtimePlotWidget 참조
        
        # 실시간 플롯 설정
        self.realtime_mode = False  # 기본값을 False로 설정
        
        # 시뮬레이션 데이터
        self.time_log = []
        self.speed_log = []
        self.roll_log = []
        self.pitch_log = []
        self.acc_log = []
        self.lidar_log = []  # LiDAR 데이터 로그 추가
        self.t = 0.0
        self.step_count = 0
        
        # 시뮬레이션 타이머
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.simulation_step)
        self.sim_interval = 50  # ms
        
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 시뮬레이션 제어 패널
        sim_control_group = QGroupBox("Simulation Control")
        sim_layout = QGridLayout()
        
        # 시뮬레이션 버튼들
        self.init_sim_btn = QPushButton("Initialize Simulation")
        self.init_sim_btn.clicked.connect(self.initialize_simulation)
        
        self.start_sim_btn = QPushButton("Start Simulation")
        self.start_sim_btn.clicked.connect(self.start_simulation)
        self.start_sim_btn.setEnabled(False)
        
        self.pause_sim_btn = QPushButton("Pause Simulation")
        self.pause_sim_btn.clicked.connect(self.pause_simulation)
        self.pause_sim_btn.setEnabled(False)
        
        self.stop_sim_btn = QPushButton("Stop Simulation")
        self.stop_sim_btn.clicked.connect(self.stop_simulation)
        self.stop_sim_btn.setEnabled(False)
        
        self.reset_sim_btn = QPushButton("Reset Simulation")
        self.reset_sim_btn.clicked.connect(self.reset_simulation)
        self.reset_sim_btn.setEnabled(False)
        
        # 실시간 모드 체크박스 추가
        self.realtime_mode_cb = QCheckBox("Real-time Plot Mode")
        self.realtime_mode_cb.setChecked(False)  # 기본값을 False로 설정
        self.realtime_mode_cb.toggled.connect(self.toggle_realtime_mode)
        
        # 시뮬레이션 설정
        self.target_speed_label = QLabel("Target Speed (km/h):")
        self.target_speed_spin = QDoubleSpinBox()
        self.target_speed_spin.setRange(0, 120)
        self.target_speed_spin.setValue(30)
        self.target_speed_spin.setSuffix(" km/h")
        
        self.sim_speed_label = QLabel("Simulation Speed:")
        self.sim_speed_spin = QDoubleSpinBox()
        self.sim_speed_spin.setRange(0.1, 5.0)
        self.sim_speed_spin.setValue(1.0)
        self.sim_speed_spin.setSingleStep(0.1)
        self.sim_speed_spin.setSuffix("x")
        self.sim_speed_spin.valueChanged.connect(self.update_sim_speed)
        
        # 레이아웃 구성
        sim_layout.addWidget(self.init_sim_btn, 0, 0)
        sim_layout.addWidget(self.start_sim_btn, 0, 1)
        sim_layout.addWidget(self.pause_sim_btn, 0, 2)
        sim_layout.addWidget(self.stop_sim_btn, 1, 0)
        sim_layout.addWidget(self.reset_sim_btn, 1, 1)
        sim_layout.addWidget(self.realtime_mode_cb, 1, 2)  # 실시간 모드 체크박스 추가
        
        sim_layout.addWidget(self.target_speed_label, 2, 0)
        sim_layout.addWidget(self.target_speed_spin, 2, 1)
        sim_layout.addWidget(self.sim_speed_label, 3, 0)
        sim_layout.addWidget(self.sim_speed_spin, 3, 1)
        
        sim_control_group.setLayout(sim_layout)
        layout.addWidget(sim_control_group)
        
        # 상태 표시
        status_group = QGroupBox("Simulation Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Status: Not Initialized")
        self.step_label = QLabel("Steps: 0")
        self.time_label = QLabel("Time: 0.00 s")
        self.speed_label = QLabel("Current Speed: 0.00 km/h")
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.step_label)
        status_layout.addWidget(self.time_label)
        status_layout.addWidget(self.speed_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # 로그 출력
        log_group = QGroupBox("Simulation Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
    def toggle_realtime_mode(self, checked):
        """실시간 모드 토글"""
        self.realtime_mode = checked
        mode_text = "enabled" if checked else "disabled"
        self.log_message(f"Real-time plot mode {mode_text}")
        
    def log_message(self, message):
        """로그 메시지 추가"""
        self.log_text.append(f"[{self.t:.2f}s] {message}")
        
    def initialize_simulation(self):
        """시뮬레이션 초기화"""
        try:
            from scripts.VehicleEnv import VehicleEnv
            import mujoco
            import mujoco.viewer
            
            self.env = VehicleEnv()
            obs = self.env.reset()
            
            # MuJoCo 뷰어 설정
            model = self.env.model
            data = self.env.data
            self.viewer = mujoco.viewer.launch_passive(model, data)
            
            # 카메라 설정
            chassis_id = model.body('chassis').id
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = chassis_id
            
            # 타겟 속도 설정
            target_speed_kmh = self.target_speed_spin.value()
            self.env.target_speed = target_speed_kmh / 3.6  # km/h to m/s
            
            # 초기 상태 설정
            self.simulation_running = False
            self.simulation_paused = False
            self.t = 0.0
            self.step_count = 0
            
            # 버튼 상태 업데이트
            self.init_sim_btn.setEnabled(False)
            self.start_sim_btn.setEnabled(True)
            self.reset_sim_btn.setEnabled(True)
            
            self.status_label.setText("Status: Initialized")
            self.log_message("Simulation initialized successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize simulation:\n{str(e)}")
            self.log_message(f"Initialization failed: {str(e)}")
            
    def start_simulation(self):
        """시뮬레이션 시작"""
        if self.env is None:
            self.log_message("Please initialize simulation first")
            return
            
        self.simulation_running = True
        self.simulation_paused = False
        self.sim_timer.start(self.sim_interval)
        
        # 버튼 상태 업데이트
        self.start_sim_btn.setEnabled(False)
        self.pause_sim_btn.setEnabled(True)
        self.stop_sim_btn.setEnabled(True)
        
        self.status_label.setText("Status: Running")
        self.log_message("Simulation started")
        
    def pause_simulation(self):
        """시뮬레이션 일시정지"""
        if self.simulation_running:
            self.simulation_paused = not self.simulation_paused
            
            if self.simulation_paused:
                self.sim_timer.stop()
                self.pause_sim_btn.setText("Resume Simulation")
                self.status_label.setText("Status: Paused")
                self.log_message("Simulation paused")
            else:
                self.sim_timer.start(self.sim_interval)
                self.pause_sim_btn.setText("Pause Simulation")
                self.status_label.setText("Status: Running")
                self.log_message("Simulation resumed")
                
    def stop_simulation(self):
        """시뮬레이션 정지"""
        was_running = self.simulation_running  # 실행 중이었는지 기록
        
        self.simulation_running = False
        self.simulation_paused = False
        self.sim_timer.stop()
        
        # 버튼 상태 업데이트
        self.start_sim_btn.setEnabled(True)
        self.pause_sim_btn.setEnabled(False)
        self.pause_sim_btn.setText("Pause Simulation")
        self.stop_sim_btn.setEnabled(False)
        
        self.status_label.setText("Status: Stopped")
        self.log_message("Simulation stopped")
        
        # 시뮬레이션이 실행 중이었고 데이터가 있으면 결과 표시
        if was_running and len(self.time_log) > 0:
            self.show_final_results()
                
    def reset_simulation(self):
        """시뮬레이션 리셋"""
        self.stop_simulation()
        
        if self.env is not None:
            obs = self.env.reset()
            
            # 데이터 초기화
            self.time_log.clear()
            self.speed_log.clear()
            self.roll_log.clear()
            self.pitch_log.clear()
            self.acc_log.clear()
            self.lidar_log.clear()  # LiDAR 로그도 초기화
            self.t = 0.0
            self.step_count = 0
            
            # RealtimePlotWidget 데이터도 초기화
            if self.realtime_widget is not None:
                self.realtime_widget.clear_data()
            
            # 타겟 속도 재설정
            target_speed_kmh = self.target_speed_spin.value()
            self.env.target_speed = target_speed_kmh / 3.6
            
            self.status_label.setText("Status: Reset")
            self.step_label.setText("Steps: 0")
            self.time_label.setText("Time: 0.00 s")
            self.speed_label.setText("Current Speed: 0.00 km/h")
            self.log_message("Simulation reset")
            
    def update_sim_speed(self, value):
        """시뮬레이션 속도 업데이트"""
        self.sim_interval = int(50 / value)  # 기본 50ms를 속도로 나눔
        if self.simulation_running and not self.simulation_paused:
            self.sim_timer.start(self.sim_interval)
            
    def simulation_step(self):
        """시뮬레이션 스텝 실행"""
        if self.env is None or not self.simulation_running:
            return
            
        try:
            # 액션 생성 (임시로 랜덤)
            action = np.random.uniform(0, 150, size=4)
            obs, reward, done, info = self.env.step(action)
            
            # 데이터 추출
            speed = self.env.data.qvel[0]  # m/s
            roll_rate = self.env.data.qvel[3]  # rad/s
            pitch_rate = self.env.data.qvel[4]  # rad/s
            vert_acc = self.env.data.qacc[2]  # m/s²
            lidar_vals = self.env.lidar
            
            # 로그 저장
            self.time_log.append(self.t)
            self.speed_log.append(speed)
            self.roll_log.append(roll_rate)
            self.pitch_log.append(pitch_rate)
            self.acc_log.append(vert_acc)
            
            # LiDAR 데이터 저장 (복사본 생성)
            if hasattr(lidar_vals, 'copy'):
                self.lidar_log.append(lidar_vals.copy())
            else:
                self.lidar_log.append(np.array(lidar_vals))
                
            self.t += self.env.sim_dt
            self.step_count += 1
            
            # 실시간 플롯 업데이트 (체크박스 상태에 따라)
            if self.realtime_mode and self.realtime_widget is not None:
                self.realtime_widget.add_data_point(
                    self.t, speed, roll_rate, pitch_rate, vert_acc, lidar_vals
                )
                
                # LiDAR 데이터 전달 디버그 (가끔씩만)
                if self.step_count % 100 == 0:
                    print(f"Step {self.step_count}: Sending LiDAR data to widget (Real-time mode)")
                    if hasattr(lidar_vals, 'shape'):
                        print(f"  LiDAR shape: {lidar_vals.shape}")
                    else:
                        print(f"  LiDAR type: {type(lidar_vals)}, length: {len(lidar_vals) if hasattr(lidar_vals, '__len__') else 'unknown'}")
                    if hasattr(lidar_vals, '__len__') and len(lidar_vals) > 0:
                        print(f"  First 5 values: {lidar_vals[:5] if len(lidar_vals) >= 5 else lidar_vals}")
            
            # MuJoCo 뷰어 동기화
            if self.viewer is not None:
                import mujoco
                mujoco.mj_step(self.env.model, self.env.data)
                self.viewer.sync()
            
            # 상태 업데이트
            self.step_label.setText(f"Steps: {self.step_count}")
            self.time_label.setText(f"Time: {self.t:.2f} s")
            self.speed_label.setText(f"Current Speed: {speed * 3.6:.2f} km/h")
            
            # 완료 체크
            if done:
                self.stop_simulation()
                self.log_message("Simulation completed!")
                # 시뮬레이션 완료 후 그래프 표시
                self.show_final_results()
                
        except Exception as e:
            self.log_message(f"Simulation error: {str(e)}")
            print(f"Simulation error details: {e}")
            self.stop_simulation()
            
    def show_final_results(self):
        """시뮬레이션 완료 후 최종 결과 그래프 표시"""
        if self.realtime_widget is not None and len(self.time_log) > 0:
            self.log_message("Displaying final simulation results...")
            
            # 실시간 모드가 비활성화된 경우에만 한 번에 모든 데이터 표시
            if not self.realtime_mode:
                # 기존 데이터 초기화
                self.realtime_widget.clear_data()
                
                # 모든 데이터를 한 번에 추가
                for i in range(len(self.time_log)):
                    lidar_data = self.lidar_log[i] if i < len(self.lidar_log) else None
                    self.realtime_widget.add_data_point(
                        self.time_log[i],
                        self.speed_log[i],
                        self.roll_log[i],
                        self.pitch_log[i],
                        self.acc_log[i],
                        lidar_data
                    )
                
                # 플롯 업데이트
                self.realtime_widget.update_plots()
                self.log_message(f"Final results displayed: {len(self.time_log)} data points")
            else:
                self.log_message("Results already displayed in real-time mode")
            
            # 결과 탭으로 자동 전환
            parent_window = self.parent()
            while parent_window and not hasattr(parent_window, 'tab_widget'):
                parent_window = parent_window.parent()
            
            if parent_window and hasattr(parent_window, 'tab_widget'):
                # Real-time Monitoring 탭으로 전환
                parent_window.tab_widget.setCurrentIndex(1)

    def closeEvent(self, event):
        """창 닫기 이벤트"""
        self.stop_simulation()
        if self.viewer is not None:
            self.viewer.close()
        if self.env is not None:
            self.env.close()
        event.accept()

    def set_realtime_widget(self, realtime_widget):
        """RealtimePlotWidget 설정"""
        self.realtime_widget = realtime_widget


class RealtimePlotWidget(QWidget):
    """실시간 플롯을 위한 메인 위젯"""
    
    def __init__(self, num_rays=32, parent=None):
        super().__init__(parent)
        self.num_rays = num_rays
        self.max_data_points = 1000  # 최대 데이터 포인트 수
        self.update_interval = 50  # ms - UI 초기화 전에 설정
        
        # 데이터 저장용 리스트
        self.time_log = []
        self.speed_log = []
        self.roll_log = []
        self.pitch_log = []
        self.acc_log = []
        self.lidar_vals = np.zeros(num_rays)
        
        self.init_ui()
        self.setup_plots()
        
        # 업데이트 타이머
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 컨트롤 패널
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 플롯 영역
        self.figure = Figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # 상태 표시줄
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.data_count_label = QLabel("Data Points: 0")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.data_count_label)
        
        status_widget = QWidget()
        status_widget.setLayout(status_layout)
        layout.addWidget(status_widget)
        
    def create_control_panel(self):
        """컨트롤 패널 생성"""
        control_group = QGroupBox("Control Panel")
        layout = QHBoxLayout()
        
        # 시작/정지 버튼
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_plotting)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_plotting)
        self.stop_btn.setEnabled(False)
        
        # 데이터 초기화 버튼
        self.clear_btn = QPushButton("Clear Data")
        self.clear_btn.clicked.connect(self.clear_data)
        
        # 저장 버튼
        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_plot)
        
        # 업데이트 간격 설정
        interval_label = QLabel("Update Interval (ms):")
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(10, 1000)
        self.interval_spin.setValue(self.update_interval)
        self.interval_spin.valueChanged.connect(self.update_interval_changed)
        
        # 최대 데이터 포인트 설정
        points_label = QLabel("Max Data Points:")
        self.points_spin = QSpinBox()
        self.points_spin.setRange(100, 10000)
        self.points_spin.setValue(self.max_data_points)
        self.points_spin.valueChanged.connect(self.max_points_changed)
        
        # 체크박스들
        self.auto_scale_cb = QCheckBox("Auto Scaling")
        self.auto_scale_cb.setChecked(True)
        self.grid_cb = QCheckBox("Show Grid")
        self.grid_cb.setChecked(True)
        self.grid_cb.toggled.connect(self.toggle_grid)
        
        # 레이아웃에 추가
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.clear_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(QLabel("|"))
        layout.addWidget(interval_label)
        layout.addWidget(self.interval_spin)
        layout.addWidget(points_label)
        layout.addWidget(self.points_spin)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.auto_scale_cb)
        layout.addWidget(self.grid_cb)
        layout.addStretch()
        
        control_group.setLayout(layout)
        return control_group
        
    def setup_plots(self):
        """플롯 설정"""
        self.figure.clear()
        
        # 5개의 서브플롯 생성
        self.axs = []
        self.axs.append(self.figure.add_subplot(5, 1, 1))  # Speed
        self.axs.append(self.figure.add_subplot(5, 1, 2))  # Roll
        self.axs.append(self.figure.add_subplot(5, 1, 3))  # Pitch  
        self.axs.append(self.figure.add_subplot(5, 1, 4))  # Acc
        self.axs.append(self.figure.add_subplot(5, 1, 5))  # LiDAR
        
        # 라인 플롯 초기화
        self.line_speed, = self.axs[0].plot([], [], 'b-', label="Speed [km/h]", linewidth=2)
        self.line_roll, = self.axs[1].plot([], [], 'r-', label="Roll Rate [rad/s]", linewidth=2)
        self.line_pitch, = self.axs[2].plot([], [], 'g-', label="Pitch Rate [rad/s]", linewidth=2)
        self.line_acc, = self.axs[3].plot([], [], 'm-', label="Vertical Acceleration [m/s²]", linewidth=2)
        
        # LiDAR 바 차트
        x_positions = np.arange(self.num_rays)
        self.bars = self.axs[4].bar(x_positions, np.zeros(self.num_rays), 
                                   color='gray', alpha=0.8, width=0.8)
        
        # 축 설정
        self.axs[0].set_ylabel("Speed [km/h]")
        self.axs[1].set_ylabel("Roll Rate [rad/s]")  
        self.axs[2].set_ylabel("Pitch Rate [rad/s]")
        self.axs[3].set_ylabel("Vertical Acceleration [m/s²]")
        self.axs[4].set_ylabel("Distance")
        self.axs[4].set_xlabel("LiDAR Ray Index")
        
        # 제목 및 범례
        self.axs[0].set_title("Vehicle and Sensor Real-time Monitoring", fontsize=14, fontweight='bold')
        for i in range(4):
            self.axs[i].legend(loc='upper right')
            self.axs[i].grid(self.grid_cb.isChecked())
            
        self.axs[4].set_ylim(0, 1.0)  # LiDAR 범위를 0-1.0으로 고정
        self.axs[4].set_xlim(-0.5, self.num_rays - 0.5)  # X축 범위 설정
        self.axs[4].grid(self.grid_cb.isChecked())
        
        # LiDAR 축에 제목 추가
        self.axs[4].set_title("LiDAR Distance (0.0 - 1.0)", fontsize=10)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        print(f"Bar chart initialized with {len(self.bars)} bars")
        
    def start_plotting(self):
        """플롯 시작"""
        self.timer.start(self.update_interval)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Running...")
        
    def stop_plotting(self):
        """플롯 정지"""
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopped")
        
    def clear_data(self):
        """데이터 초기화"""
        self.time_log.clear()
        self.speed_log.clear()
        self.roll_log.clear()
        self.pitch_log.clear()
        self.acc_log.clear()
        self.lidar_vals = np.zeros(self.num_rays)
        self.update_plots()
        self.data_count_label.setText("Data Points: 0")
        
    def save_plot(self):
        """플롯 이미지 저장"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "simulation_results.png", 
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        if filename:
            try:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                QMessageBox.information(self, "Save Complete", f"Plot saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error occurred while saving:\n{str(e)}")
                
    def update_interval_changed(self, value):
        """업데이트 간격 변경"""
        self.update_interval = value
        if self.timer.isActive():
            self.timer.start(value)
            
    def max_points_changed(self, value):
        """최대 데이터 포인트 수 변경"""
        self.max_data_points = value
        
    def toggle_grid(self, checked):
        """격자 표시 토글"""
        for ax in self.axs:
            ax.grid(checked)
        self.canvas.draw()
        
    def add_data_point(self, time_val, speed, roll, pitch, acc, lidar_data=None):
        """새로운 데이터 포인트 추가"""
        self.time_log.append(time_val)
        self.speed_log.append(speed)
        self.roll_log.append(roll)
        self.pitch_log.append(pitch)
        self.acc_log.append(acc)
        
        # LiDAR 데이터 처리 개선
        if lidar_data is not None:
            if hasattr(lidar_data, 'shape'):
                # NumPy 배열인 경우
                if len(lidar_data.shape) == 1:
                    # 1D 배열 - 직접 사용
                    self.lidar_vals = np.array(lidar_data[:self.num_rays])
                elif len(lidar_data.shape) == 2:
                    # 2D 배열 - 첫 번째 행 사용 또는 평균
                    self.lidar_vals = np.array(lidar_data[0][:self.num_rays])
                else:
                    print(f"Warning: Unexpected LiDAR data shape: {lidar_data.shape}")
                    self.lidar_vals = np.zeros(self.num_rays)
            elif isinstance(lidar_data, (list, tuple)):
                # 리스트/튜플인 경우
                self.lidar_vals = np.array(lidar_data[:self.num_rays])
            else:
                print(f"Warning: Unknown LiDAR data type: {type(lidar_data)}")
                self.lidar_vals = np.zeros(self.num_rays)
                
            # LiDAR 데이터 디버그 (가끔씩만)
            if len(self.time_log) % 50 == 0:  # 더 자주 출력 (50스텝마다)
                print(f"LiDAR Bar Chart Update: count={len(self.lidar_vals)}, "
                      f"min={np.min(self.lidar_vals):.3f}, "
                      f"max={np.max(self.lidar_vals):.3f}, "
                      f"mean={np.mean(self.lidar_vals):.3f}")
                # 강제로 플롯 업데이트
                self.update_plots()
        else:
            # LiDAR 데이터가 없으면 기본값 유지
            pass
            
        # 최대 데이터 포인트 수 제한
        if len(self.time_log) > self.max_data_points:
            self.time_log.pop(0)
            self.speed_log.pop(0)
            self.roll_log.pop(0)
            self.pitch_log.pop(0)
            self.acc_log.pop(0)
            
        self.data_count_label.setText(f"Data Points: {len(self.time_log)}")
        
        # 데이터가 추가되면 즉시 플롯 업데이트
        self.update_plots()
        
        # 타이머가 실행 중이 아니면 자동으로 시작
        if not self.timer.isActive():
            self.timer.start(self.update_interval)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("Running...")
            
    def update_plots(self):
        """플롯 업데이트"""
        if not self.time_log:
            return
            
        # 시계열 데이터 업데이트
        time_array = np.array(self.time_log)
        self.line_speed.set_data(time_array, np.array(self.speed_log) * 3.6)  # m/s → km/h
        self.line_roll.set_data(time_array, self.roll_log)
        self.line_pitch.set_data(time_array, self.pitch_log)
        self.line_acc.set_data(time_array, self.acc_log)
        
        # 자동 스케일링
        if self.auto_scale_cb.isChecked():
            for i in range(4):
                self.axs[i].relim()
                self.axs[i].autoscale_view()
                
        # LiDAR 바 차트 업데이트 (고정 범위 0-1.0)
        if len(self.lidar_vals) > 0:
            # 가끔씩만 디버그 출력
            if len(self.time_log) % 100 == 0:
                print(f"Updating LiDAR bars: {len(self.lidar_vals)} values, range: {np.min(self.lidar_vals):.3f}-{np.max(self.lidar_vals):.3f}")
            
            # 바 차트 높이 업데이트
            updated_count = 0
            for i, (bar, val) in enumerate(zip(self.bars, self.lidar_vals)):
                if i < len(self.lidar_vals):
                    bar.set_height(val)
                    updated_count += 1
                    # 거리에 따른 색상 변경 (0-1.0 범위)
                    if val < 0.2:  # 0.2 이내 - 매우 가까움
                        bar.set_color('red')
                    elif val < 0.4:  # 0.4 이내 - 가까움
                        bar.set_color('orange')
                    elif val < 0.6:  # 0.6 이내 - 보통
                        bar.set_color('yellow')
                    elif val < 0.8:  # 0.8 이내 - 멀음
                        bar.set_color('lightblue')
                    else:  # 0.8 이상 - 매우 멀음
                        bar.set_color('cyan')
            
            # 가끔씩만 업데이트 카운트 출력
            if len(self.time_log) % 100 == 0:
                print(f"Updated {updated_count} bars")
            
            # LiDAR 축 범위를 0-1.0으로 고정
            self.axs[4].set_ylim(0, 1.0)
        elif len(self.time_log) % 100 == 0:  # 데이터가 없을 때도 가끔씩만 출력
            print("No LiDAR data to update bars")
            
        self.canvas.draw()


class MainPlotWindow(QMainWindow):
    """메인 플롯 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("Vehicle Suspension Simulation - Real-time Monitoring")
        self.setGeometry(100, 100, 1400, 900)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 탭 위젯
        self.tab_widget = QTabWidget()  # self.tab_widget으로 변경하여 인스턴스 변수로 만듦
        
        # 실시간 플롯 탭 (먼저 생성)
        self.realtime_widget = RealtimePlotWidget()
        
        # 시뮬레이션 제어 탭 (RealtimePlotWidget 참조 전달)
        self.simulation_controller = SimulationController(realtime_widget=self.realtime_widget)
        
        self.tab_widget.addTab(self.simulation_controller, "Simulation Control")
        
        # 실시간 플롯 탭 추가
        self.tab_widget.addTab(self.realtime_widget, "Real-time Monitoring")
        
        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.tab_widget)
        central_widget.setLayout(layout)
        
        # 상태 표시줄
        self.statusBar().showMessage("Ready - Please initialize simulation first")
        
    def closeEvent(self, event):
        """창 닫기 이벤트"""
        # 시뮬레이션 정리
        if hasattr(self, 'simulation_controller'):
            self.simulation_controller.closeEvent(event)
        event.accept()


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