# config/bump_config.py

from dataclasses import dataclass, asdict

@dataclass
class BumpConfig:
    
    seed: int = 42

    num_bumps: int = 5

    a_range: tuple = (2.5, 3.5)
    # b_range: tuple = (0.3, 0.5)
    b_range: tuple = (0.1, 0.2)
    h_range: tuple = (4.5, 5.0)
    segments: int = 80

    x_start: float = 5.0
    min_gap: float = 5.0
    max_gap: float = 10.0

    def to_dict(self):
        return asdict(self)

DEFAULT_BUMP_CONFIG = BumpConfig()
