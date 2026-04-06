import base64
import io

from PIL import Image

from config import CROP_SIZE_RATIO

MAX_IMAGE_BYTES = 3_750_000


def encode_image(pil_image: Image.Image) -> tuple[str, str]:
    """Encode a PIL image to base64, staying under the API size limit.

    Tries PNG first; falls back to JPEG with decreasing quality.
    """
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    if buf.tell() <= MAX_IMAGE_BYTES:
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/png"

    img_rgb = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
    for quality in (95, 85, 75, 60):
        buf = io.BytesIO()
        img_rgb.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= MAX_IMAGE_BYTES:
            return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"

    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=40)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def crop_around_point(
    pil_image: Image.Image,
    nx: float,
    ny: float,
    crop_ratio: float = CROP_SIZE_RATIO,
    resize_to_original: bool = True,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop a region centred on (nx, ny) and optionally resize back to original dims."""
    w, h = pil_image.size
    px, py = nx * w, ny * h

    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)

    left = max(0, int(px - crop_w // 2))
    upper = max(0, int(py - crop_h // 2))
    right = min(w, left + crop_w)
    lower = min(h, upper + crop_h)

    if right - left < crop_w and left > 0:
        left = max(0, right - crop_w)
    if lower - upper < crop_h and upper > 0:
        upper = max(0, lower - crop_h)

    crop = pil_image.crop((left, upper, right, lower))
    if resize_to_original:
        crop = crop.resize((w, h), Image.LANCZOS)

    return crop, (left, upper, right, lower)


def map_zoomin_prediction(
    nx: float,
    ny: float,
    crop_bbox: tuple[int, int, int, int],
    orig_w: int,
    orig_h: int,
) -> tuple[float, float]:
    """Map a normalised prediction from the zoomed view back to original image space."""
    left, upper, right, lower = crop_bbox
    orig_x = left + nx * (right - left)
    orig_y = upper + ny * (lower - upper)
    return orig_x / orig_w, orig_y / orig_h
