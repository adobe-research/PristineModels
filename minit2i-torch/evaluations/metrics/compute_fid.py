from torch import Tensor
from torchmetrics.image.fid import FrechetInceptionDistance


class FIDScorer:
    def __init__(self, **kwargs):
        self._m = FrechetInceptionDistance(**kwargs)

    def to(self, *args, **kwargs):
        self._m.to(*args, **kwargs)
        return self

    def update(self, real_images: Tensor, fake_images: Tensor) -> None:
        self._m.update(real_images, real=True)
        self._m.update(fake_images, real=False)

    def compute(self) -> float:
        return float(self._m.compute())

    def reset(self) -> None:
        self._m.reset()
