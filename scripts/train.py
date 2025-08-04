# scripts/train_vehicle_env.py

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from scripts.VehicleEnv import VehicleEnv
from scripts.callbacks import EpisodeLoggerCallback

def train():
    env = VehicleEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    callback = EpisodeLoggerCallback()

    print("================[Model Training Start]================")
    model.learn(total_timesteps=1000, callback=callback)

    # 3. 모델 저장
    model.save("scripts/ppo_vehicle_model")
    print("================[Model Training Completed]================")

if __name__ == "__main__":
    train()
