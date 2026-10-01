import io
import os
import tempfile
import warnings
from pathlib import Path

from PIL import Image, ImageOps

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


def read_image(raw: bytes) -> Image.Image:
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image must contain 1 byte to 10 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds 25 million pixels")
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Use JPEG, PNG or WebP")
                image.verify()
            with Image.open(io.BytesIO(raw)) as image:
                return ImageOps.exif_transpose(image).convert("RGB")
    except (
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError("Invalid or oversized image") from error


def atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
