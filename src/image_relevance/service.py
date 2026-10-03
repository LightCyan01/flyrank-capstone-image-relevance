import hashlib
import json
import uuid
from pathlib import Path

from image_relevance.config import Settings
from image_relevance.matching import cosine, guard
from image_relevance.models import atomic_write, read_image
from image_relevance.schemas import (
    BatchInput,
    MatchReviewInput,
    Metadata,
    PostInput,
    ReviewInput,
    valid_vector,
)
from image_relevance.store import Store


class ConflictError(ValueError):
    pass


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def public_image(row: dict) -> dict:
    fields = (
        "id",
        "filename",
        "status",
        "metadata",
        "vision_model",
        "caption_model",
        "embedding_model",
    )
    result = {key: row[key] for key in fields}
    result["metadata"] = json.loads(row["metadata"]) if row["metadata"] else None
    return result


class Service:
    def __init__(self, store: Store, settings: Settings):
        self.store = store
        self.settings = settings

    def add_image(self, tenant: str, filename: str, raw: bytes) -> dict:
        read_image(raw).close()
        checksum = digest(raw)
        self.store.tenant(tenant)
        existing = self.store.rows(
            "SELECT * FROM images WHERE tenant_id=? AND digest=?", (tenant, checksum)
        )
        if existing:
            return public_image(existing[0])
        # Opaque paths prevent client filenames or tenant names becoming filesystem paths.
        tenant_folder = digest(tenant.encode())[:32]
        path = self.settings.data_dir / "images" / tenant_folder / checksum
        atomic_write(path, raw)
        resource_id = uuid.uuid4().hex
        safe_name = Path(filename.replace("\\", "/")).name[:200] or "image"
        with self.store.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO images(id,tenant_id,digest,filename,path) VALUES(?,?,?,?,?)",
                (resource_id, tenant, checksum, safe_name, str(path.resolve())),
            )
        return public_image(
            self.store.rows(
                "SELECT * FROM images WHERE tenant_id=? AND digest=?", (tenant, checksum)
            )[0]
        )

    def add_post(self, tenant: str, post: PostInput) -> dict:
        self.store.tenant(tenant)
        checksum = digest(post.model_dump_json().encode())
        with self.store.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO posts(id,tenant_id,title,content,digest) VALUES(?,?,?,?,?)",
                (uuid.uuid4().hex, tenant, post.title, post.content, checksum),
            )
            row = db.execute(
                "SELECT id,title,content,status FROM posts WHERE tenant_id=? AND digest=?",
                (tenant, checksum),
            ).fetchone()
        return dict(row)

    def enqueue(
        self, tenant: str, request_key: str, batch: BatchInput, *, retry_failed: bool = False
    ) -> dict:

        if not batch.image_ids and not batch.post_ids:
            raise ValueError("A batch needs at least one image or article")
        if not 1 <= len(request_key) <= 100 or not request_key.isascii():
            raise ValueError("Idempotency-Key must be 1 to 100 ASCII characters")
        payload = {"images": sorted(batch.image_ids)}
        if batch.post_ids:
            payload["posts"] = sorted(batch.post_ids)
        payload_hash = digest(json.dumps(payload, sort_keys=True).encode())

        self.store.tenant(tenant)

        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM jobs WHERE tenant_id=? AND request_key=?", (tenant, request_key)
            ).fetchone()
            if existing:
                if existing["payload_hash"] != payload_hash:
                    raise ConflictError("Idempotency-Key was already used for another batch")

                if retry_failed and existing["status"] == "failed":
                    db.execute(
                        "UPDATE job_items SET status='queued',attempts=0,retry_at=0,error=NULL "
                        "WHERE job_id=? AND status='failed'",
                        (existing["id"],),
                    )
                    db.execute("UPDATE jobs SET status='queued' WHERE id=?", (existing["id"],))
                    db.commit()
                return self.job(tenant, existing["id"])

            for table, ids in (("images", batch.image_ids), ("posts", batch.post_ids)):
                for resource_id in ids:
                    if not db.execute(
                        f"SELECT 1 FROM {table} WHERE tenant_id=? AND id=?", (tenant, resource_id)
                    ).fetchone():
                        raise LookupError("Resource not found")

            job_id = uuid.uuid4().hex

            db.execute(
                "INSERT INTO jobs(id,tenant_id,request_key,payload_hash) VALUES(?,?,?,?)",
                (job_id, tenant, request_key, payload_hash),
            )

            db.executemany(
                "INSERT INTO job_items(tenant_id,job_id,image_id) VALUES(?,?,?)",
                [(tenant, job_id, image_id) for image_id in batch.image_ids],
            )
            db.executemany(
                "INSERT INTO job_items(tenant_id,job_id,post_id) VALUES(?,?,?)",
                [(tenant, job_id, post_id) for post_id in batch.post_ids],
            )
        return self.job(tenant, job_id)

    def job(self, tenant: str, job_id: str) -> dict:
        row = self.store.get("jobs", tenant, job_id)

        items = self.store.rows(
            "SELECT item.image_id,item.post_id,image.filename,post.title,"
            "item.status,item.attempts,item.error FROM job_items AS item "
            "LEFT JOIN images AS image ON image.id=item.image_id "
            "AND image.tenant_id=item.tenant_id "
            "LEFT JOIN posts AS post ON post.id=item.post_id AND post.tenant_id=item.tenant_id "
            "WHERE item.tenant_id=? "
            "AND item.job_id=? ORDER BY item.id",
            (tenant, job_id),
        )

        return {
            "id": row["id"],
            "status": row["status"],
            "total": len(items),
            "completed": sum(item["status"] == "done" for item in items),
            "failed": sum(item["status"] == "failed" for item in items),
            "items": items,
        }

    def start_batch(
        self, tenant: str, image_ids: list[str], post_ids: list[str] | None = None
    ) -> dict:
        batch = BatchInput(image_ids=sorted(set(image_ids)), post_ids=sorted(set(post_ids or [])))
        version = [
            batch.model_dump(),
            self.settings.vision_model,
            self.settings.caption_model,
            self.settings.embedding_model,
        ]
        request_key = digest(json.dumps(version).encode())
        return self.enqueue(tenant, request_key, batch, retry_failed=True)

    def costs(self, tenant: str, limit: int = 100, offset: int = 0) -> dict:
        calls = self.store.rows(
            "SELECT call.*,item.image_id,item.post_id FROM model_calls AS call "
            "JOIN job_items AS item ON item.id=call.item_id "
            "WHERE call.tenant_id=? ORDER BY call.id LIMIT ? OFFSET ?",
            (tenant, limit, offset),
        )

        totals = self.store.rows(
            "SELECT COUNT(*) AS calls,COALESCE(SUM(cost_usd),0) AS cost_usd "
            "FROM model_calls WHERE tenant_id=?",
            (tenant,),
        )[0]

        return {
            "provider": "local",
            "totals": totals,
            "calls": calls,
            "daily_call_limit": self.settings.daily_call_limit,
        }

    def start_match(self, tenant: str, post: PostInput, image_ids: list[str]) -> dict:
        if not image_ids:
            image_ids = [
                row["id"]
                for row in self.store.rows(
                    "SELECT id FROM images WHERE tenant_id=? ORDER BY rowid", (tenant,)
                )
            ]
        if not image_ids:
            raise ValueError("Choose at least one image, or run uv run image-relevance demo first")
        image_ids = sorted(set(image_ids))
        BatchInput(image_ids=image_ids)
        post_id = self.add_post(tenant, post)["id"]
        job = self.start_batch(tenant, image_ids, [post_id])
        with self.store.connect() as db:
            db.execute("UPDATE tenants SET latest_match_id=? WHERE id=?", (job["id"], tenant))
        return {
            "match_id": job["id"],
            "status": job["status"],
            "next_step": "Open step 2: GET /matches/latest. Execute again while processing.",
            "status_url": f"/matches/{job['id']}",
        }

    def match_result(self, tenant: str, match_id: str) -> dict:
        job = self.job(tenant, match_id)
        posts = [item["post_id"] for item in job["items"] if item["post_id"]]
        images = [item["image_id"] for item in job["items"] if item["image_id"]]
        if len(posts) != 1 or not images:
            raise ValueError("Use a match_id returned by POST /matches")
        result = {
            "match_id": match_id,
            "status": job["status"],
            "processed": job["completed"],
            "total": job["total"],
        }
        if job["status"] != "completed":
            result["errors"] = [item["error"] for item in job["items"] if item["error"]]
            return result
        ranked = self.rank(tenant, posts[0], image_ids=images)
        numbers = {image_id: number for number, image_id in enumerate(images, 1)}
        for candidate in ranked["candidates"]:
            candidate["image_number"] = numbers[candidate["image_id"]]
            candidate["image_url"] = f"/images/{candidate['image_id']}/file"
        result.update(
            status=ranked["status"],
            suggestions=ranked["suggestions"][:5],
            rejected_images=len(ranked["candidates"]) - len(ranked["suggestions"]),
            reasons=ranked["reasons"],
        )
        result["next_step"] = "Review a chosen image in step 3 using match_id and image_number."
        return result

    def review_match(self, tenant: str, match_id: str, review: MatchReviewInput) -> dict:
        job = self.job(tenant, match_id)
        posts = [item["post_id"] for item in job["items"] if item["post_id"]]
        images = [item["image_id"] for item in job["items"] if item["image_id"]]
        if job["status"] != "completed" or len(posts) != 1 or review.image_number > len(images):
            raise ValueError("That image_number is not available; check the match result first")
        ranked = self.rank(tenant, posts[0], images[review.image_number - 1])
        if "suggestion_id" not in ranked["candidates"][0]:
            raise ConflictError("Image is not ready; submit the article again")
        return self.review(
            tenant,
            ranked["candidates"][0]["suggestion_id"],
            ReviewInput(decision=review.decision, note=review.note),
        )

    def rank(
        self, tenant: str, post_id: str, image_id: str | None = None,
        *, image_ids: list[str] | None = None,
    ) -> dict:
        post = self.store.get("posts", tenant, post_id)
        if post["status"] != "ready" or not post["vector"]:
            raise ConflictError("Article is not ready; check its processing job")
        if post["embedding_model"] != self.settings.embedding_model:
            raise ConflictError("Embedding model changed; submit this article again")
        post_vector = valid_vector(json.loads(post["vector"]))
        if image_id:
            images = [self.store.get("images", tenant, image_id)]
        elif image_ids is not None:
            images = [self.store.get("images", tenant, value) for value in image_ids]
        else:
            images = self.store.rows(
                "SELECT * FROM images WHERE tenant_id=? ORDER BY rowid", (tenant,)
            )
        candidates = []
        with self.store.connect() as db:
            for image in images:
                if not image["metadata"] or not image["vector"]:
                    candidates.append(
                        {
                            "image_id": image["id"],
                            "filename": image["filename"],
                            "allowed": False,
                            "similarity": None,
                            "explanation": ["Image is not ready; process it in an image batch"],
                        }
                    )
                    continue
                metadata = Metadata.model_validate_json(image["metadata"])
                score = cosine(post_vector, valid_vector(json.loads(image["vector"])))
                reasons = guard(
                    post["title"] + "\n" + post["content"],
                    metadata,
                    score,
                    self.settings.min_confidence,
                    self.settings.min_similarity,
                )
                if image["status"] not in {"ready", "needs_review"}:
                    reasons.append("Image processing is incomplete or failed; check its job")
                for column, current in (
                    ("vision_model", self.settings.vision_model),
                    ("caption_model", self.settings.caption_model),
                    ("embedding_model", self.settings.embedding_model),
                ):
                    if image[column] != current:
                        reasons.append(f"{column.replace('_', ' ')} changed; reprocess this image")
                explanation = reasons or [
                    f"Subject agrees ({metadata.subject}); similarity {score:.3f} and "
                    f"vision confidence {metadata.confidence:.3f} clear both thresholds"
                ]
                row = db.execute(
                    "INSERT INTO suggestions(id,tenant_id,post_id,image_id,similarity,allowed,"
                    "explanation) VALUES(?,?,?,?,?,?,?) ON CONFLICT(tenant_id,post_id,image_id) "
                    "DO UPDATE SET similarity=excluded.similarity,allowed=excluded.allowed,"
                    "explanation=excluded.explanation RETURNING id,decision",
                    (
                        uuid.uuid4().hex,
                        tenant,
                        post_id,
                        image["id"],
                        score,
                        int(not reasons),
                        json.dumps(explanation),
                    ),
                ).fetchone()
                candidates.append(
                    {
                        "suggestion_id": row["id"],
                        "decision": row["decision"],
                        "image_id": image["id"],
                        "filename": image["filename"],
                        "caption": metadata.caption,
                        "attributes": metadata.attributes,
                        "subject": metadata.subject,
                        "similarity": round(score, 6),
                        "confidence": metadata.confidence,
                        "allowed": not reasons,
                        "explanation": explanation,
                    }
                )

        candidates.sort(
            key=lambda candidate: (
                candidate["similarity"] if candidate["similarity"] is not None else -2
            ),
            reverse=True,
        )
        suggestions = [
            candidate for candidate in candidates
            if candidate["allowed"] and candidate["decision"] != "rejected"
        ]
        reasons = []
        if not suggestions:
            if not candidates:
                reasons.append("No images are available; upload an image batch first")
            for candidate in candidates:
                if candidate.get("decision") == "rejected":
                    reasons.append(f"Reviewer rejected {candidate['filename']}")
                elif not candidate["allowed"]:
                    reasons.extend(candidate["explanation"])
        return {
            "post_id": post_id,
            "status": "suggested" if suggestions else "no_confident_match",
            "suggestions": suggestions,
            "candidates": candidates,
            "reasons": list(dict.fromkeys(reasons)),
        }

    def suggestion(self, tenant: str, suggestion_id: str) -> dict:
        row = self.store.get("suggestions", tenant, suggestion_id)
        row["explanation"] = json.loads(row["explanation"])
        row["allowed"] = bool(row["allowed"])
        return row

    def review(self, tenant: str, suggestion_id: str, review: ReviewInput) -> dict:
        row = self.store.get("suggestions", tenant, suggestion_id)
        if review.decision == "approved":
            # Check current models and thresholds before accepting an earlier suggestion.
            ranked = self.rank(tenant, row["post_id"], row["image_id"])
            if not ranked["candidates"][0]["allowed"]:
                raise ConflictError("Cannot approve a pairing rejected by the mismatch guard")
        with self.store.connect() as db:
            db.execute(
                "UPDATE suggestions SET decision=?,note=?,reviewed_at=CURRENT_TIMESTAMP "
                "WHERE tenant_id=? AND id=?",
                (review.decision, review.note, tenant, suggestion_id),
            )
        return self.suggestion(tenant, suggestion_id)
