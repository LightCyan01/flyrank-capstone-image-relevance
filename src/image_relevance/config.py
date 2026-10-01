import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def load_env(path: Path = Path(".env")) -> None:
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                key, separator, value = line.partition("=")
                if not separator or not key.replace("_", "").isalnum():
                    raise ValueError("Invalid .env entry")
                os.environ.setdefault(key, value.strip().strip("\"'"))


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    device: str = "cpu"
    vision_model: str = "yolo26s-cls.pt"
    caption_model: str = "Salesforce/blip-image-captioning-base"
    min_confidence: float = 0.55
    daily_call_limit: int = 500
    api_keys: dict[str, str] = field(default_factory=dict)
    retry_delay: float = 1.0

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        return cls(
            data_dir=Path(os.getenv("IMAGE_RELEVANCE_DATA_DIR", "data")),
            device=os.getenv("IMAGE_RELEVANCE_DEVICE", cls.device),
            vision_model=os.getenv("IMAGE_RELEVANCE_VISION_MODEL", cls.vision_model),
            caption_model=os.getenv("IMAGE_RELEVANCE_CAPTION_MODEL", cls.caption_model),
            min_confidence=float(os.getenv("IMAGE_RELEVANCE_MIN_CONFIDENCE", "0.55")),
            daily_call_limit=int(os.getenv("IMAGE_RELEVANCE_DAILY_CALL_LIMIT", "500")),
            api_keys=json.loads(os.getenv("IMAGE_RELEVANCE_API_KEYS", "{}")),
        )

    def __post_init__(self) -> None:
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        if self.daily_call_limit < 1 or self.retry_delay < 0:
            raise ValueError("Invalid call budget or retry delay")
        if (
            not isinstance(self.api_keys, dict)
            or any(
                not isinstance(k, str) or not k or not isinstance(v, str) or len(v) < 16
                for k, v in self.api_keys.items()
            )
            or len(set(self.api_keys.values())) != len(self.api_keys)
        ):
            raise ValueError("API keys must be unique tokens of at least 16 characters")
