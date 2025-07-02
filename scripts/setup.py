# scripts/setup.py

from utils import create_new_scene, BumpConfig

def main():
    # Config 설정
    bump_cfg = BumpConfig()

    # Bump 생성 
    create_new_scene("models/scenes/base_scene.xml", "models/scenes/new_scene.xml", bump_cfg)

if __name__ == "__main__":
    main()
