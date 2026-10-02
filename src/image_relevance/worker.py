import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from filelock import FileLock

from image_relevance.config import Settings
from image_relevance.models import Models
from image_relevance.schemas import Metadata, valid_vector
from image_relevance.store import Store

logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 3


class BudgetError(ValueError):
    pass


class Worker:
    def __init__(self, store: Store, settings: Settings, models: Models | None = None):
        self.store = store
        self.settings = settings
        self.models = models or Models(settings)
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = FileLock(settings.data_dir / "worker.lock", timeout=0)

    def start(self) -> None:
        self.lock.acquire()
        try:
            self.stopping.clear()
            self.recover()
            self.thread = threading.Thread(
                target=self.run, name="image-relevance-worker", daemon=True
            )
            self.thread.start()
        except Exception:
            self.lock.release()
            raise

    def stop(self) -> None:
        self.stopping.set()
        if self.thread is not None:
            self.thread.join()
        self.lock.release()

    def recover(self) -> None:
        with self.store.connect() as db:
            db.execute(
                "UPDATE model_calls SET status='interrupted',error='Worker restarted' "
                "WHERE status='started'"
            )
            db.execute("UPDATE job_items SET status='queued' WHERE status='running'")
            db.execute("UPDATE jobs SET status='queued' WHERE status='running'")

    def claim(self) -> dict | None:
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            item = db.execute(
                "SELECT * FROM job_items WHERE status='queued' AND retry_at<=? ORDER BY id LIMIT 1",
                (time.time(),),
            ).fetchone()
            if item is None:
                return None
            item = dict(item)
            db.execute(
                "UPDATE job_items SET status='running',attempts=attempts+1 WHERE id=?",
                (item["id"],),
            )
            db.execute("UPDATE jobs SET status='running' WHERE id=?", (item["job_id"],))
            table = "images" if item["image_id"] else "posts"
            resource_id = item["image_id"] or item["post_id"]
            db.execute(
                f"UPDATE {table} SET status='processing' WHERE tenant_id=? AND id=?",
                (item["tenant_id"], resource_id),
            )
            item["attempts"] += 1
            return item

    def call(self, item: dict, kind: str, model: str, operation, *args) -> Any:
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "SELECT COUNT(*) FROM model_calls WHERE tenant_id=? AND created_at>=date('now')",
                (item["tenant_id"],),
            ).fetchone()[0]
            if count >= self.settings.daily_call_limit:
                raise BudgetError("Daily local model-call budget exceeded")
            call_id = db.execute(
                "INSERT INTO model_calls(tenant_id,job_id,item_id,kind,attempt,model) "
                "VALUES(?,?,?,?,?,?)",
                (
                    item["tenant_id"],
                    item["job_id"],
                    item["id"],
                    kind,
                    item["attempts"],
                    model,
                ),
            ).lastrowid
        started = time.perf_counter()
        status, message = "succeeded", None
        try:
            result = operation(*args)
            return valid_vector(result) if kind == "embedding" else Metadata.model_validate(result)
        except Exception as error:
            # Model errors may contain signed URLs or secrets; record only their class.
            status, message = "failed", type(error).__name__
            raise
        finally:
            with self.store.connect() as db:
                db.execute(
                    "UPDATE model_calls SET status=?,duration_ms=?,error=? WHERE id=?",
                    (status, (time.perf_counter() - started) * 1000, message, call_id),
                )

    def describe_image(self, item: dict, row: dict) -> Metadata:
        tenant, image_id = item["tenant_id"], item["image_id"]
        if (
            row["metadata"]
            and row["vision_model"] == self.settings.vision_model
            and row["caption_model"] == self.settings.caption_model
        ):
            return Metadata.model_validate_json(row["metadata"])
        path = Path(row["path"])
        classification = self.call(
            item, "classification", self.settings.vision_model, self.models.classify, path
        )
        metadata = self.call(
            item, "caption", self.settings.caption_model, self.models.describe, path, classification
        )
        tags = [("subject", metadata.subject), ("category", metadata.category)]
        tags += [("attribute", value) for value in metadata.attributes]
        with self.store.connect() as db:
            db.execute(
                "UPDATE images SET metadata=?,vision_model=?,caption_model=?,"
                "vector=NULL,embedding_model=NULL "
                "WHERE tenant_id=? AND id=?",
                (
                    metadata.model_dump_json(),
                    self.settings.vision_model,
                    self.settings.caption_model,
                    tenant,
                    image_id,
                ),
            )
            db.execute("DELETE FROM tags WHERE tenant_id=? AND image_id=?", (tenant, image_id))
            db.executemany(
                "INSERT OR IGNORE INTO tags VALUES(?,?,?,?)",
                [(tenant, image_id, kind, value) for kind, value in tags],
            )
        row["vector"] = None
        return metadata

    def process(self, item: dict) -> None:
        tenant = item["tenant_id"]
        table = "images" if item["image_id"] else "posts"
        resource_id = item["image_id"] or item["post_id"]
        row = self.store.get(table, tenant, resource_id)
        status = "ready"
        if item["image_id"]:
            metadata = self.describe_image(item, row)
            text = metadata.caption
            if metadata.confidence < self.settings.min_confidence or metadata.category != "animal":
                status = "needs_review"
        else:
            text = row["title"] + "\n" + row["content"]
        if not row["vector"] or row["embedding_model"] != self.settings.embedding_model:
            vector = self.call(
                item, "embedding", self.settings.embedding_model, self.models.embed, text
            )
            with self.store.connect() as db:
                db.execute(
                    f"UPDATE {table} SET vector=?,embedding_model=? WHERE tenant_id=? AND id=?",
                    (json.dumps(vector), self.settings.embedding_model, tenant, resource_id),
                )
        with self.store.connect() as db:
            db.execute(
                f"UPDATE {table} SET status=? WHERE tenant_id=? AND id=?",
                (status, tenant, resource_id),
            )

    def finish(self, item: dict, error: Exception | None = None) -> None:
        status, message = "done", None
        if error is not None:
            status = "queued"
            if isinstance(error, (ValueError, LookupError)) or item["attempts"] >= MAX_ATTEMPTS:
                status = "failed"
            message = f"Processing failed ({type(error).__name__})"
            if isinstance(error, BudgetError):
                message = str(error)
        with self.store.connect() as db:
            db.execute(
                "UPDATE job_items SET status=?,retry_at=?,error=? WHERE id=?",
                (
                    status,
                    time.time() + self.settings.retry_delay * 2 ** item["attempts"],
                    message,
                    item["id"],
                ),
            )
            if status == "failed":
                table = "images" if item["image_id"] else "posts"
                db.execute(
                    f"UPDATE {table} SET status='failed' WHERE tenant_id=? AND id=?",
                    (item["tenant_id"], item["image_id"] or item["post_id"]),
                )
                db.execute(
                    "INSERT OR IGNORE INTO alerts(tenant_id,job_id,item_id,message) "
                    "VALUES(?,?,?,?)",
                    (item["tenant_id"], item["job_id"], item["id"], message),
                )
            unfinished = db.execute(
                "SELECT COUNT(*) FROM job_items WHERE job_id=? AND status IN ('queued','running')",
                (item["job_id"],),
            ).fetchone()[0]
            if not unfinished:
                failures = db.execute(
                    "SELECT COUNT(*) FROM job_items WHERE job_id=? AND status='failed'",
                    (item["job_id"],),
                ).fetchone()[0]
                db.execute(
                    "UPDATE jobs SET status=? WHERE id=?",
                    ("failed" if failures else "completed", item["job_id"]),
                )
        if status == "failed":
            logger.error("Background item failed; inspect GET /alerts (job=%s)", item["job_id"])

    def run(self) -> None:
        while not self.stopping.is_set():
            item = self.claim()
            if item is None:
                self.stopping.wait(0.2)
                continue
            try:
                if item["attempts"] > MAX_ATTEMPTS:
                    raise ValueError("Interrupted retry limit exceeded")
                self.process(item)
            except Exception as error:
                self.finish(item, error)
            else:
                self.finish(item)
