import json
from datetime import UTC, datetime

from image_relevance.config import Settings
from image_relevance.corpus import CORPUS_DIR
from image_relevance.matching import subjects
from image_relevance.models import atomic_write
from image_relevance.service import Service, digest
from image_relevance.store import Store


def load_seed(settings: Settings, tenant: str) -> dict:
    name = digest(tenant.encode())[:16]
    path = settings.data_dir / f"seed-{name}.json"
    if not path.is_file():
        raise ValueError("Run uv run image-relevance seed first")
    return json.loads(path.read_text(encoding="utf-8"))


def measure_posts(service: Service, tenant: str, seed: dict, posts: list[dict]) -> dict:
    image_names = {image_id: name for name, image_id in seed["images"].items()}
    results = []

    for post in posts:
        ranked = service.rank(tenant, seed["posts"][post["key"]], image_ids=list(image_names))
        # Benchmark model choices; a review decision must not change the measured accuracy.
        allowed = [candidate for candidate in ranked["candidates"] if candidate["allowed"]]
        predicted_image = None
        similarity = None
        subject_correct = False
        if allowed:
            best = allowed[0]
            predicted_image = image_names.get(best["image_id"], best["image_id"])
            similarity = best["similarity"]
            expected_subject = subjects(post["title"] + " " + post["content"])
            subject_correct = subjects(best["subject"]) == expected_subject
        results.append(
            {
                "post": post["key"],
                "expected_image": post["correct_image"],
                "predicted_image": predicted_image,
                "correct": predicted_image == post["correct_image"],
                "subject_correct": subject_correct,
                "similarity": similarity,
            }
        )

    total = len(results)
    correct = sum(result["correct"] for result in results)
    return {
        "posts": total,
        "correct": correct,
        "top1_precision": correct / total,
        "subject_precision": sum(result["subject_correct"] for result in results) / total,
        "results": results,
    }


def evaluate(settings: Settings, tenant: str = "demo") -> dict:
    seed = load_seed(settings, tenant)
    service = Service(Store(settings.data_dir), settings)
    posts = json.loads((CORPUS_DIR / "posts.json").read_text(encoding="utf-8"))
    calibration = [post for post in posts if post["split"] == "calibration"]
    labeled = [post for post in posts if post["split"] == "eval"]
    if len(labeled) < 10 or not calibration:
        raise ValueError("Evaluation needs ten labeled posts and separate calibration examples")
    image_keys = set(seed["images"])
    for post in calibration + labeled:
        if post["correct_image"] not in image_keys:
            raise ValueError(f"Unknown labeled image for {post['key']}")

    # Check the chosen thresholds on separate examples before scoring the evaluation posts.
    calibration_result = measure_posts(service, tenant, seed, calibration)
    report = {
        "measured_at": datetime.now(UTC).isoformat(),
        **measure_posts(service, tenant, seed, labeled),
        "definition": (
            "First model-allowed image matches the label; abstentions count as incorrect. "
            "Only seeded photos are ranked, and human decisions do not affect this benchmark."
        ),
        "vision_model": settings.vision_model,
        "caption_model": settings.caption_model,
        "embedding_model": settings.embedding_model,
        "min_confidence": settings.min_confidence,
        "min_similarity": settings.min_similarity,
        "calibration": calibration_result,
    }
    atomic_write(settings.data_dir / "evaluation.json", json.dumps(report, indent=2).encode())
    return report
