import hashlib
import io
import tempfile
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from PIL import Image
from pydantic import ValidationError

from image_relevance.corpus import download_image, load_images, verify_image
from image_relevance.schemas import Metadata


def check_metadata() -> None:
    example = {
        "subject": "red fox",
        "category": "animal",
        "attributes": [" orange fur ", "forest"],
        "caption": "A red fox standing in a forest",
        "confidence": 0.94,
    }
    metadata = Metadata.model_validate(example)
    assert metadata.attributes == ["orange fur", "forest"]

    for changed in (
        {"subject": " "},
        {"caption": ""},
        {"attributes": [" "]},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"confidence": float("nan")},
        {"confidence": "0.94"},
        {"unexpected": "field"},
    ):
        try:
            Metadata.model_validate({**example, **changed})
        except ValidationError:
            continue
        raise AssertionError(f"Invalid metadata accepted: {changed}")
    print("Metadata accepts valid output and rejects invalid fields and confidence")


def check_downloads() -> None:
    image_bytes = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(image_bytes, format="JPEG")
    raw = image_bytes.getvalue()
    entry = {"key": "test-01", "url": "https://example.invalid/test.jpg"}
    entry["sha256"] = hashlib.sha256(raw).hexdigest()

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        path = directory / "test-01.jpg"
        path.write_bytes(b"interrupted download")
        with patch("image_relevance.corpus.urlopen", return_value=io.BytesIO(raw)) as request:
            assert download_image(entry, directory).read_bytes() == raw
            assert download_image(entry, directory).read_bytes() == raw
            assert request.call_count == 1
        changed = {**entry, "sha256": "0" * 64}
        with patch("image_relevance.corpus.urlopen", return_value=io.BytesIO(raw)):
            try:
                download_image(changed, directory)
            except ValueError:
                pass
            else:
                raise AssertionError("Changed source checksum was accepted")
        assert path.read_bytes() == raw
        assert list(directory.iterdir()) == [path]
    print("Downloads repair partial files, reuse verified photos and refuse changed sources")


def check_corpus() -> None:
    images = load_images()
    groups = {entry["subject"] for entry in images}
    assert len(images) >= 40 and len(groups) >= 4
    assert len({entry["key"] for entry in images}) == len(images)
    assert len({entry["sha256"] for entry in images}) == len(images)
    for entry in images:
        assert entry["key"] == Path(entry["key"]).name
        assert entry["author"] and entry["license"]
        assert entry["source"].startswith("https://commons.wikimedia.org/")
        assert urlsplit(entry["license_url"]).scheme in {"http", "https"}
        extension = Path(urlsplit(entry["url"]).path).suffix.lower()
        verify_image((Path("data/corpus") / f"{entry['key']}{extension}").read_bytes(), entry)
    print(f"Verified {len(images)} licensed photos across {len(groups)} animal groups")


if __name__ == "__main__":
    check_metadata()
    check_downloads()
    check_corpus()
    print("PHASE 1 CHECK PASSED")
