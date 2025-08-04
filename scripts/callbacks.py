# scripts/callbacks.py

from stable_baselines3.common.callbacks import BaseCallback

class EpisodeLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_num = 0
        self.episode_reward = 0.0
        self.episode_length = 0

    def _on_step(self) -> bool:
        # 현재 step의 reward 누적
        self.episode_reward += self.locals['rewards'][0]
        self.episode_length += 1

        # 에피소드 종료 시 출력
        if self.locals['dones'][0]:
            self.episode_num += 1
            print(f"===============[Episode {self.episode_num} finished]===============")
            print(f"   - Total Reward   : {self.episode_reward:.2f}")
            print(f"   - Episode Length : {self.episode_length} steps")

            # 다음 에피소드 초기화
            self.episode_reward = 0.0
            self.episode_length = 0

        return True
