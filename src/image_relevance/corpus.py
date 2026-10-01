import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from image_relevance.models import MAX_IMAGE_BYTES, atomic_write, read_image

MANIFEST = Path(__file__).resolve().parents[2] / "corpus" / "images.json"


def load_images() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def verify_image(raw: bytes, entry: dict) -> None:
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"Invalid image size: {entry['key']}")

    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"Image checksum changed: {entry['key']}")

    read_image(raw).close()


def download_image(entry: dict, directory: Path) -> Path:
    extension = Path(urlsplit(entry["url"]).path).suffix.lower()
    path = directory / f"{entry['key']}{extension}"
    raw = path.read_bytes() if path.is_file() else b""
    needs_download = hashlib.sha256(raw).hexdigest() != entry["sha256"]

    if needs_download:
        request = Request(
            entry["url"],
            headers={
                "User-Agent": (
                    "Image-Relevance/0.1"
                    "(https://github.com/LightCyan01/flyrank-capstone-image-relevance)"
                )
            },
        )
        with urlopen(request, timeout=45) as response:
            raw = response.read(MAX_IMAGE_BYTES + 1)
    verify_image(raw, entry)

    if needs_download:
        atomic_write(path, raw)
    return path


def main() -> None:
    images = load_images()
    for entry in images:
        path = download_image(entry, Path("data/corpus"))
        print(f"Ready: {path.name}")
    groups = {entry["subject"] for entry in images}
    print(f"Prepared {len(images)} photos across {len(groups)} animal groups")


if __name__ == "__main__":
    main()
