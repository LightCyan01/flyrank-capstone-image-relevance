CREATE TABLE tenants (id TEXT PRIMARY KEY);

CREATE TABLE images (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL REFERENCES tenants(id),
    digest TEXT NOT NULL, filename TEXT NOT NULL, path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', metadata TEXT,
    vision_model TEXT, caption_model TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (tenant_id, digest), UNIQUE (tenant_id, id)
);
CREATE INDEX images_tenant_status ON images(tenant_id, status);

CREATE TABLE tags (
    tenant_id TEXT NOT NULL, image_id TEXT NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL,
    PRIMARY KEY (tenant_id, image_id, kind, value),
    FOREIGN KEY (tenant_id, image_id) REFERENCES images(tenant_id, id) ON DELETE CASCADE
);
CREATE INDEX tags_lookup ON tags(tenant_id, kind, value);

CREATE TABLE jobs (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL REFERENCES tenants(id),
    request_key TEXT NOT NULL, payload_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (tenant_id, request_key), UNIQUE (tenant_id, id)
);

CREATE TABLE job_items (
    id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, job_id TEXT NOT NULL, image_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
    retry_at REAL NOT NULL DEFAULT 0, error TEXT,
    UNIQUE (job_id, image_id), UNIQUE (tenant_id, job_id, id),
    FOREIGN KEY (tenant_id, job_id) REFERENCES jobs(tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, image_id) REFERENCES images(tenant_id, id)
);
CREATE INDEX items_queue ON job_items(status, retry_at);

CREATE TABLE model_calls (
    id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, job_id TEXT NOT NULL, item_id INTEGER NOT NULL,
    kind TEXT NOT NULL, attempt INTEGER NOT NULL, model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'started', cost_usd REAL NOT NULL DEFAULT 0 CHECK(cost_usd=0),
    duration_ms REAL, error TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tenant_id, job_id, item_id) REFERENCES job_items(tenant_id, job_id, id)
);
CREATE INDEX calls_budget ON model_calls(tenant_id, created_at);

CREATE TABLE alerts (
    id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, job_id TEXT NOT NULL,
    item_id INTEGER NOT NULL UNIQUE, message TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tenant_id, job_id, item_id) REFERENCES job_items(tenant_id, job_id, id)
);
