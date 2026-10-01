import argparse
import logging
import secrets
from pathlib import Path

from image_relevance.config import Settings
from image_relevance.models import atomic_write


def init() -> None:
    path = Path(".env")
    if not path.exists():
        example = Path(".env.example").read_text(encoding="utf-8")
        atomic_write(
            path,
            example.replace("replace-with-a-long-random-token", secrets.token_urlsafe(32)).encode(),
        )
        print("Created .env with a random local API token. Keep this file private.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local YOLO image understanding")
    parser.add_argument("command", nargs="?", default="serve", choices=["serve", "demo", "seed"])
    parser.add_argument("--tenant", default="demo")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Libraries emit their own progress; suppress request logs that may contain signed URLs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    init()
    settings = Settings.from_env()
    if args.command in {"serve", "demo"}:
        import uvicorn

        from image_relevance.api import create_app

        if args.command == "demo":
            from image_relevance.corpus import seed

            print("Preparing demo images and local models. The first run downloads model weights.")
            seed(settings, args.tenant)
        print(f"Open http://127.0.0.1:{args.port}/docs to upload photos and inspect their tags.")
        print("Authorize with your demo token from IMAGE_RELEVANCE_API_KEYS in .env.")
        uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, workers=1)
    elif args.command == "seed":
        from image_relevance.corpus import seed

        seed(settings, args.tenant)


if __name__ == "__main__":
    main()
