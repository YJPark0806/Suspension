import sys
sys.path.append(".")     # 현재 디렉토리를 PYTHONPATH에 추가

import scripts           # __init__.py가 실행되어 환경이 등록됨
import gym

# Gym 환경 생성
env = gym.make("VehicleSuspension-v0", render_mode="human")
obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
