# scripts/train_vehicle_env.py

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from scripts.VehicleEnv import VehicleEnv

def train_ppo_model():
    # 1. 환경 생성
    env = VehicleEnv()


    # 2. 모델 생성 및 학습
    model = PPO("MlpPolicy", env, verbose=1)
    print("================[Model Training Start]================")
    model.learn(total_timesteps=1000)

    # 3. 모델 저장
    model.save("scripts/ppo_vehicle_model")
    print("================[Model Training Completed]================")

if __name__ == "__main__":
    train_ppo_model()
