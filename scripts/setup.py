# scripts/setup.py

from config import BumpConfig, SceneConfig
from utils import add_bumps, generate_random_bumps, save_bumps
from pathlib import Path

def main():
    # Config 설정
    scene_cfg = SceneConfig()
    bump_cfg = BumpConfig()

    # Bump 생성 
    bumps = generate_random_bumps(bump_cfg.to_dict())
    save_bumps(bumps, scene_cfg.bump_dir)
    add_bumps(scene_cfg, bump_cfg)

if __name__ == "__main__":
    main()
