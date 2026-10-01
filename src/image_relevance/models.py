import io
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import cast

from PIL import Image, ImageOps

from image_relevance.config import Settings
from image_relevance.schemas import Metadata

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


class Models:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.vision = None
        self.captioner = None

    def classify(self, path: Path) -> Metadata:
        from ultralytics import YOLO
        from ultralytics.engine.results import Results

        if self.vision is None:
            model = Path(self.settings.vision_model)
            if not model.is_absolute():
                model = Path("models") / model
            model.parent.mkdir(parents=True, exist_ok=True)
            loaded = YOLO(str(model))
            if loaded.task != "classify":
                raise ValueError("Use a YOLO classification model (*-cls.pt)")
            self.vision = loaded
        with read_image(path.read_bytes()) as image:
            results = cast(
                list[Results],
                self.vision.predict(image, device=self.settings.device, verbose=False),
            )
        result = results[0]
        if result.probs is None:
            raise ValueError("Classification model returned no probabilities")
        label = result.names[result.probs.top1].replace("_", " ")
        detected = re.search(r"\b(fox|wolf|dog|bear|deer|elk|grizzly)\b", label) is not None
        # ImageNet dog breed names do not contain the word 'dog'.
        imagenet_labels = (
            len(result.names) == 1000 and result.names.get(151, "").lower() == "chihuahua"
        )
        if not detected and imagenet_labels and 151 <= result.probs.top1 <= 268:
            label = f"{label} dog"
            detected = True
        return Metadata(
            subject=label,
            category="animal" if detected else "other",
            attributes=[],
            caption=f"A photograph of a {label}.",
            confidence=float(result.probs.top1conf),
        )

    def describe(self, path: Path, metadata: Metadata) -> Metadata:
        from torch.nn import Module
        from transformers import BlipForConditionalGeneration, BlipProcessor

        if self.captioner is None:
            cache = str(self.settings.data_dir / "model-cache")
            processor = BlipProcessor.from_pretrained(self.settings.caption_model, cache_dir=cache)
            model = cast(
                BlipForConditionalGeneration,
                BlipForConditionalGeneration.from_pretrained(
                    self.settings.caption_model, cache_dir=cache
                ),
            )
            cast(Module, model).to(self.settings.device)
            model.eval()
            self.captioner = (processor, model)
        processor, model = self.captioner
        with read_image(path.read_bytes()) as image:
            inputs = processor(
                images=image,
                text=f"a {metadata.subject}",
                text_kwargs={"return_tensors": "pt"},
                images_kwargs={"return_tensors": "pt"},
            ).to(self.settings.device)
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        caption = processor.decode(output[0], skip_special_tokens=True)
        # Keep only details the caption model actually named
        details = (
            "red",
            "orange",
            "brown",
            "white",
            "black",
            "gray",
            "snow",
            "grass",
            "forest",
            "water",
            "rocks",
            "sitting",
            "standing",
            "walking",
            "running",
            "lying",
            "tree",
        )
        attributes = [detail for detail in details if re.search(rf"\b{detail}\b", caption)][:10]
        return Metadata.model_validate(
            {**metadata.model_dump(), "caption": caption, "attributes": attributes}
        )
