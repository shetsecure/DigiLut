import openslide
from pathlib import Path


class WholeSlideImage:
    def __init__(self, path: Path):
        path = Path(path)
        self.name = path.stem

        self.wsi = openslide.open_slide(str(path))
        self.level_downsamples = self._assert_level_downsamples()
        self.level_dim = self.wsi.level_dimensions

    def _assert_level_downsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]

        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            (
                level_downsamples.append(estimated_downsample)
                if estimated_downsample != (downsample, downsample)
                else level_downsamples.append((downsample, downsample))
            )

        return level_downsamples
