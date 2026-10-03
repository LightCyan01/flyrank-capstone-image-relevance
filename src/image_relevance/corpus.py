import hashlib
import io
import json
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from filelock import Timeout
from PIL import Image

from image_relevance.config import Settings
from image_relevance.models import MAX_IMAGE_BYTES, Models, atomic_write, read_image
from image_relevance.schemas import PostInput
from image_relevance.service import Service, digest
from image_relevance.store import Store
from image_relevance.worker import Worker

CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus"
MANIFEST = CORPUS_DIR / "images.json"

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


def seed(settings: Settings, tenant: str = "demo", *, models: Models | None = None) -> dict:
    if tenant not in settings.api_keys:
        raise ValueError("Tenant must be configured in IMAGE_RELEVANCE_API_KEYS")
    store = Store(settings.data_dir)
    service = Service(store, settings)
    images = {}
    for entry in load_images():
        path = download_image(entry, settings.data_dir / "corpus")
        images[entry["key"]] = service.add_image(tenant, path.name, path.read_bytes())["id"]
    # A blank image checks how the pipeline handles an unsupported subject.
    stream = io.BytesIO()
    Image.new("RGB", (256, 256), "gray").save(stream, format="PNG")
    images["uncertain-01"] = service.add_image(tenant, "uncertain-01.png", stream.getvalue())["id"]
    posts = {}
    for post in json.loads((CORPUS_DIR / "posts.json").read_text(encoding="utf-8")):
        posts[post["key"]] = service.add_post(
            tenant, PostInput(title=post["title"], content=post["content"])
        )["id"]
    job = service.start_batch(tenant, list(images.values()), list(posts.values()))
    worker = Worker(store, settings, models)
    try:
        worker.start()
    except Timeout:
        # The running server owns the same queue and will process this batch.
        pass
    try:
        deadline = time.monotonic() + 1800
        previous_completed = -1
        while time.monotonic() < deadline:
            job = service.job(tenant, job["id"])
            if job["completed"] != previous_completed:
                print(f"Processed {job['completed']}/{job['total']} items", flush=True)
                previous_completed = job["completed"]
            if job["status"] == "failed":
                raise RuntimeError("Batch failed; inspect /jobs and /alerts")
            if job["status"] == "completed":
                mapping = {"images": images, "posts": posts, "job_id": job["id"]}
                name = digest(tenant.encode())[:16]
                atomic_write(
                    settings.data_dir / f"seed-{name}.json",
                    json.dumps(mapping, indent=2).encode(),
                )
                print(json.dumps({
                    "seeded_images": len(images), "seeded_posts": len(posts),
                    "status": job["status"], **service.costs(tenant)["totals"],
                }))
                return job
            time.sleep(0.2)
        raise TimeoutError("Batch is still pending; progress is saved in the database")
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
