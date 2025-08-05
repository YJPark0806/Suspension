import sys
from PyQt5.QtWidgets import QApplication
from utils.plot import MainPlotWindow

def main():
    """GUI 기반 시뮬레이션 애플리케이션 실행"""
    # PyQt 애플리케이션 생성
    app = QApplication(sys.argv)
    
    # 메인 윈도우 생성 및 표시
    main_window = MainPlotWindow()
    main_window.show()
    
    print("=== Vehicle Suspension Simulation ===")
    print("GUI Application Started!")
    print("Please use the 'Simulation Control' tab to:")
    print("1. Initialize Simulation")
    print("2. Start/Stop/Pause Simulation")
    print("3. Monitor data in 'Real-time Monitoring' tab")
    print("")
    print("📊 Plot Mode:")
    print("- Real-time Plot Mode: OFF (default) - 시뮬레이션 완료 후 그래프 표시")
    print("- Real-time Plot Mode: ON - 실시간으로 그래프 업데이트")
    print("- 체크박스를 통해 모드 변경 가능")
    print("=====================================")
    
    # 이벤트 루프 실행
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()