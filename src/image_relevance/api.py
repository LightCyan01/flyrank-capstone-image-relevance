import hmac
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image

from image_relevance.config import Settings
from image_relevance.models import MAX_IMAGE_BYTES
from image_relevance.schemas import BatchInput, PostInput
from image_relevance.service import ConflictError, Service, public_image
from image_relevance.store import Store
from image_relevance.worker import Worker


def create_app(settings: Settings | None = None, *, models=None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = Store(settings.data_dir)
    service = Service(store, settings)
    worker = Worker(store, settings, models)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not settings.api_keys:
            raise RuntimeError("Set IMAGE_RELEVANCE_API_KEYS or start with uv run image-relevance")
        worker.start()
        try:
            yield
        finally:
            worker.stop()

    app = FastAPI(
        title="Image Relevance",
        lifespan=lifespan,
        description=(
            "Local YOLO classification and BLIP captions. Click **Authorize** and paste your "
            "token from IMAGE_RELEVANCE_API_KEYS in .env.\n\n"
            "**1. Upload** photos with POST /batches. **2. Check progress** with GET /jobs. "
            "**3. See captions and tags** with GET /images.\n\n"
            "To try the included photos, run `uv run image-relevance demo`. "
            "Images marked needs_review have low confidence or an unsupported subject."
            "\n\n**Match an article:** submit its title and text with POST /posts, "
            "then open the returned results_url for ranked images and rejection reasons."
        ),
        openapi_tags=[
            {"name": "Process images", "description": "Upload, check progress, inspect results."},
            {
                "name": "Match an article",
                "description": "Submit an article, then check its results.",
            },
            {"name": "Diagnostics", "description": "Per-call costs and processing failures."},
            {"name": "Advanced", "description": "Individual images and explicit batches."},
        ],
        swagger_ui_parameters={"docExpansion": "list", "defaultModelsExpandDepth": -1},
    )
    app.state.store = store
    app.state.service = service
    app.state.worker = worker
    security = HTTPBearer(auto_error=False)

    def authenticate(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    ) -> str:
        if credentials is not None:
            for tenant, token in settings.api_keys.items():
                if hmac.compare_digest(credentials.credentials.encode(), token.encode()):
                    return tenant
        raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(LookupError)
    async def not_found(request, error):
        return JSONResponse(status_code=404, content={"detail": "Resource not found"})

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse(
            status_code=409 if isinstance(error, ConflictError) else 400,
            content={"detail": str(error)},
        )

    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/docs")

    @app.post("/batches", status_code=202, tags=["Process images"], summary="1. Upload photos")
    def batch(
        files: Annotated[
            list[UploadFile],
            File(
                min_length=1,
                max_length=200,
                description="JPEG, PNG or WebP photos",
                json_schema_extra={"items": {"type": "string", "format": "binary"}},
            ),
        ],
        tenant: Annotated[str, Depends(authenticate)],
    ):
        image_ids = []
        for file in files:
            raw = file.file.read(MAX_IMAGE_BYTES + 1)
            image_ids.append(service.add_image(tenant, file.filename or "image", raw)["id"])
        job = service.start_batch(tenant, image_ids)
        return {**job, "status_url": f"/jobs/{job['id']}", "results_url": "/images"}

    @app.get("/jobs", tags=["Process images"], summary="2. Check batch progress")
    def jobs(tenant: Annotated[str, Depends(authenticate)]):
        recent = store.rows(
            "SELECT id FROM jobs WHERE tenant_id=? ORDER BY rowid DESC LIMIT 20", (tenant,)
        )
        return [service.job(tenant, job["id"]) for job in recent]

    @app.get("/images", tags=["Process images"], summary="3. See captions and tags")
    def images(
        tenant: Annotated[str, Depends(authenticate)],
        limit: Annotated[int, Query(ge=1, le=200)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        return [
            public_image(row)
            for row in store.rows(
                "SELECT * FROM images WHERE tenant_id=? ORDER BY rowid LIMIT ? OFFSET ?",
                (tenant, limit, offset),
            )
        ]

    @app.get("/costs", tags=["Diagnostics"], summary="Inspect model calls and costs")
    def costs(
        tenant: Annotated[str, Depends(authenticate)],
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        return service.costs(tenant, limit, offset)

    @app.post("/posts", status_code=202, tags=["Match an article"], summary="1. Submit an article")
    def add_post(post: PostInput, tenant: Annotated[str, Depends(authenticate)]):
        saved = service.add_post(tenant, post)
        images = store.rows("SELECT id FROM images WHERE tenant_id=? ORDER BY rowid", (tenant,))
        if len(images) > 200:
            raise ValueError("The article workflow supports a library of up to 200 images")
        job = service.start_batch(tenant, [image["id"] for image in images], [saved["id"]])
        return {
            "post_id": saved["id"],
            "job_id": job["id"],
            "status": job["status"],
            "status_url": f"/jobs/{job['id']}",
            "results_url": f"/posts/{saved['id']}/images",
        }

    @app.get("/posts", tags=["Match an article"], summary="Find submitted articles")
    def posts(
        tenant: Annotated[str, Depends(authenticate)],
        limit: Annotated[int, Query(ge=1, le=200)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        return store.rows(
            "SELECT id,title,content,status FROM posts WHERE tenant_id=? "
            "ORDER BY rowid DESC LIMIT ? OFFSET ?",
            (tenant, limit, offset),
        )

    @app.get(
        "/posts/{post_id}/images",
        tags=["Match an article"],
        summary="2. See ranked images and reasons",
    )
    def rank(post_id: str, tenant: Annotated[str, Depends(authenticate)]):
        return service.rank(tenant, post_id)

    @app.get(
        "/posts/{post_id}/images/{image_id}", tags=["Advanced"], summary="Check a specific image"
    )
    def candidate(post_id: str, image_id: str, tenant: Annotated[str, Depends(authenticate)]):
        return service.rank(tenant, post_id, image_id)

    @app.get("/alerts", tags=["Diagnostics"])
    def alerts(tenant: Annotated[str, Depends(authenticate)]):
        return store.rows(
            "SELECT * FROM alerts WHERE tenant_id=? ORDER BY id DESC LIMIT 100", (tenant,)
        )

    @app.get("/health", tags=["Diagnostics"])
    def health():
        if worker.thread is None or not worker.thread.is_alive():
            raise HTTPException(503, "Background worker is not running")
        return {"status": "ok"}

    @app.post("/images", status_code=201, tags=["Advanced"])
    def add_image(
        file: Annotated[UploadFile, File(json_schema_extra={"format": "binary"})],
        tenant: Annotated[str, Depends(authenticate)],
    ):
        return service.add_image(
            tenant, file.filename or "image", file.file.read(MAX_IMAGE_BYTES + 1)
        )

    @app.get("/images/{image_id}", tags=["Advanced"])
    def image(image_id: str, tenant: Annotated[str, Depends(authenticate)]):
        return public_image(store.get("images", tenant, image_id))

    @app.get("/images/{image_id}/file", tags=["Advanced"])
    def image_file(image_id: str, tenant: Annotated[str, Depends(authenticate)]):
        row = store.get("images", tenant, image_id)
        with Image.open(row["path"]) as original:
            media_type = Image.MIME[original.format or "PNG"]
        return FileResponse(row["path"], media_type=media_type)

    @app.post("/jobs", status_code=202, tags=["Advanced"])
    def enqueue(
        batch: BatchInput,
        tenant: Annotated[str, Depends(authenticate)],
        idempotency_key: Annotated[str, Header(min_length=1, max_length=100)],
    ):
        return service.enqueue(tenant, idempotency_key, batch)

    @app.get("/jobs/{job_id}", tags=["Advanced"])
    def job(job_id: str, tenant: Annotated[str, Depends(authenticate)]):
        return service.job(tenant, job_id)

    return app
