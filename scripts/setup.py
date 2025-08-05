# scripts/setup.py

from utils import create_all_bumps, BumpConfig

def main():
    bump_cfg = BumpConfig(
        num_bumps=200  # STL 개수 늘림
    )

    # 전체 bump STL 및 XML 반영 준비
    create_all_bumps(
        base_scene_path="models/scenes/base_scene.xml",
        output_scene_path="models/scenes/new_scene.xml",
        bump_config=bump_cfg
    )

if __name__ == "__main__":
    main()
