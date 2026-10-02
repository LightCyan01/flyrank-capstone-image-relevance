import hashlib
import json
import uuid
from pathlib import Path

from image_relevance.config import Settings
from image_relevance.models import atomic_write, read_image
from image_relevance.schemas import BatchInput, PostInput
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
            "LEFT JOIN images AS image ON image.id=item.image_id AND image.tenant_id=item.tenant_id "
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
            batch.model_dump(), self.settings.vision_model,
            self.settings.caption_model, self.settings.embedding_model,
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
