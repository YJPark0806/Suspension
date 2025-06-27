# # scripts/debug.py

# # index 확인용
# from scripts.VehicleEnv import VehicleEnv
# from mujoco import MjModel

# def get_joint_names(model: MjModel):
#     names = []
#     for i in range(model.njnt):
#         addr = model.name_jntadr[i]
#         name = model.names[addr:].split(b'\x00', 1)[0].decode('utf-8')
#         names.append(name)
#     return names

# def main():
#     env = VehicleEnv()
#     model = env.model

#     print("=== Joint Index Mapping ===")
#     print(f"nq (qpos size): {model.nq}")
#     print(f"nv (qvel size): {model.nv}")
#     print()

#     joint_names = get_joint_names(model)
#     for i, name in enumerate(joint_names):
#         qpos_id = model.jnt_qposadr[i]
#         qvel_id = model.jnt_dofadr[i]
#         print(f"{name:20} → qpos[{qpos_id}], qvel[{qvel_id}]")

#     print("\nNote: First 7 elements of qpos are base position + orientation (x, y, z, qw, qx, qy, qz)")
#     print("      First 6 elements of qvel are base linear + angular velocity (vx, vy, vz, wx, wy, wz)")

#     env.close()

# if __name__ == "__main__":
#     main()

# 단위 확인용 출력
# scripts/debug.py

import time
from scripts.VehicleEnv import VehicleEnv

def main():
    # 환경 초기화
    env = VehicleEnv()
    model = env.model
    data = env.data

    obs = env.reset()

    print("\n--- fl_wheel 상태 추적 시작 (단위: rad, rad/s) ---\n")

    try:
        for step in range(100):
            # 일정한 속도로 전진
            action = [env.target_speed, 0.0, 0, 0, 0, 0]
            obs, reward, done, info = env.step(action)

            angle = data.qpos[5]
            angvel = data.qvel[5]
            print(f"step {step:03d} | fl_wheel angle: {angle:.4f} rad | angular vel: {angvel:.4f} rad/s")

            time.sleep(env.sim_dt)

    except KeyboardInterrupt:
        print("강제 종료")
    finally:
        env.close()

if __name__ == "__main__":
    main()
