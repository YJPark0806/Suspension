from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

def default_out_xml(scene_dir: Path) -> Path:
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return scene_dir / f"scene_{now_str}.xml"

@dataclass
class SceneConfig:
    scene_dir: Path = Path("models/scenes")
    base_xml: Path = field(init=False)
    out_xml: Path = field(init=False)
    bump_dir: Path = field(init=False)
    bumps: list = field(default_factory=list)
    default_rgba: str = "1 1 0 1"

    def __post_init__(self):
        self.base_xml = self.scene_dir / "base_scene.xml"
        self.out_xml = default_out_xml(self.scene_dir)
        self.bump_dir = self.scene_dir.parent / "speed_bumps"

DEFAULT_SCENE_CONFIG = SceneConfig()
