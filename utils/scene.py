# utils/scene.py

import re
import mujoco

from pathlib import Path

from config.scene_config import DEFAULT_SCENE_CONFIG

def find_latest_scene(scene_dir: Path) -> Path | None:
    pattern = re.compile(r"scene_(\d{8}_\d{6})\.xml$")
    candidates = []
    for f in scene_dir.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                candidates.append((m.group(1), f))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def get_scene_path(
    scene_path: Path | None = None, 
    scene_dir: Path = DEFAULT_SCENE_CONFIG.scene_dir
) -> Path | None:
    if scene_path is None:
        scene_path = find_latest_scene(scene_dir)
    if scene_path is None:
        raise FileNotFoundError(f"No scene file found in {scene_dir}")
    return str(scene_path.resolve())
    
def load_scene(scene_path: Path | None = None):
    scene_path = get_scene_path(scene_path, Path("models/scenes"))
    model = mujoco.MjModel.from_xml_path(str(scene_path.resolve()))
    data = mujoco.MjData(model)
    return model, data
