from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareConfig:
    """Target accelerator parameters used by the Roofline model."""

    name: str = "Example GPU"
    peak_flops: float = 15e12
    peak_memory_bandwidth: float = 900e9

    @property
    def ridge_point(self) -> float:
        return self.peak_flops / self.peak_memory_bandwidth


DEFAULT_HARDWARE = HardwareConfig()
