from dataclasses import dataclass

from PIL import Image


@dataclass
class ScreenSpotSample:
    idx: int
    file_name: str
    image: Image.Image
    instruction: str
    bbox: list[float]
    data_type: str
    data_source: str

    @property
    def img_size(self) -> tuple[int, int]:
        return self.image.size


def parse_normal(idx: int, row: dict) -> ScreenSpotSample:
    return ScreenSpotSample(
        idx=idx,
        file_name=row.get("file_name", f"sample_{idx}"),
        image=row["image"],
        instruction=row["instruction"],
        bbox=row["bbox"],
        data_type=row.get("data_type", "unknown"),
        data_source=row.get("data_source", row.get("data_souce", "unknown")),
    )


def parse_pro(idx: int, row: dict) -> ScreenSpotSample:
    return ScreenSpotSample(
        idx=idx,
        file_name=row.get("img_filename", f"sample_{idx}"),
        image=row["image"],
        instruction=row["instruction"],
        bbox=row["bbox"],
        data_type=row.get("ui_type", "unknown"),
        data_source=row.get("application", row.get("platform", "unknown")),
    )


PARSERS = {
    "normal": parse_normal,
    "pro": parse_pro,
}
