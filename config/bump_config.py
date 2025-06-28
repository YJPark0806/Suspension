# config/bump_config.py

from dataclasses import dataclass, asdict

@dataclass
class BumpConfig:
    
    seed: int = 42

    num_bumps: int = 5

    a_range: tuple = (2.5, 3.5)
    b_range: tuple = (0.1, 0.2)
    h_range: tuple = (4.5, 5.0)
    segments: int = 80

    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0

    def to_dict(self):
        return asdict(self)

DEFAULT_BUMP_CONFIG = BumpConfig()
