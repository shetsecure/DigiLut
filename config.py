from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class Config:
    seg_level: int = 5
    sthresh: int = 20
    sthresh_up: int = 255
    mthresh: int = 7
    close: int = 0
    use_otsu: bool = False
    filter_params: Dict[str, Any] = field(default_factory=lambda: {"a_t": 2, "a_h": 10, "max_n_holes": 9})
    ref_patch_size: int = 512

    # for filtering thin contours
    min_aspect_ratio: float = 0.3
    min_area: int = 2e5
