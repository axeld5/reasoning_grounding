from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    hf_name: str
    split: str
    streaming: bool


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "normal": DatasetConfig(
        hf_name="rootsautomation/ScreenSpot",
        split="test",
        streaming=False,
    ),
    "pro": DatasetConfig(
        hf_name="lscpku/ScreenSpot-Pro",
        split="test",
        streaming=False,
    ),
}
