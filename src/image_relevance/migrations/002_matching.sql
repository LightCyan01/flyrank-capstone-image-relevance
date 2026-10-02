ALTER TABLE images ADD COLUMN vector TEXT;
ALTER TABLE images ADD COLUMN embedding_model TEXT;

CREATE TABLE posts (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL REFERENCES tenants(id),
    title TEXT NOT NULL, content TEXT NOT NULL, digest TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', vector TEXT, embedding_model TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (tenant_id, digest), UNIQUE (tenant_id, id)
);
CREATE INDEX posts_tenant_status ON posts(tenant_id, status);

-- SQLite needs a replacement table to make image_id nullable for post jobs.
-- Copy dependent rows before dropping the old tables so call history survives.
CREATE TABLE job_items_new (
    id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, job_id TEXT NOT NULL,
    image_id TEXT, post_id TEXT,
    status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
    retry_at REAL NOT NULL DEFAULT 0, error TEXT,
    CHECK ((image_id IS NOT NULL) != (post_id IS NOT NULL)),
    UNIQUE (job_id, image_id), UNIQUE (job_id, post_id), UNIQUE (tenant_id, job_id, id),
    FOREIGN KEY (tenant_id, job_id) REFERENCES jobs(tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, image_id) REFERENCES images(tenant_id, id),
    FOREIGN KEY (tenant_id, post_id) REFERENCES posts(tenant_id, id)
);
INSERT INTO job_items_new(id,tenant_id,job_id,image_id,status,attempts,retry_at,error)
SELECT id,tenant_id,job_id,image_id,status,attempts,retry_at,error FROM job_items;

CREATE TABLE model_calls_new (
    id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, job_id TEXT NOT NULL, item_id INTEGER NOT NULL,
    kind TEXT NOT NULL, attempt INTEGER NOT NULL, model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'started', cost_usd REAL NOT NULL DEFAULT 0 CHECK(cost_usd=0),
    duration_ms REAL, error TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tenant_id, job_id, item_id) REFERENCES job_items_new(tenant_id, job_id, id)
);
INSERT INTO model_calls_new SELECT * FROM model_calls;

CREATE TABLE alerts_new (
    id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, job_id TEXT NOT NULL,
    item_id INTEGER NOT NULL UNIQUE, message TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tenant_id, job_id, item_id) REFERENCES job_items_new(tenant_id, job_id, id)
);
INSERT INTO alerts_new SELECT * FROM alerts;

DROP TABLE alerts;
DROP TABLE model_calls;
DROP TABLE job_items;
ALTER TABLE job_items_new RENAME TO job_items;
ALTER TABLE model_calls_new RENAME TO model_calls;
ALTER TABLE alerts_new RENAME TO alerts;
CREATE INDEX items_queue ON job_items(status, retry_at);
CREATE INDEX calls_budget ON model_calls(tenant_id, created_at);

CREATE TABLE suggestions (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, post_id TEXT NOT NULL, image_id TEXT NOT NULL,
    similarity REAL NOT NULL, allowed INTEGER NOT NULL, explanation TEXT NOT NULL,
    UNIQUE (tenant_id, post_id, image_id),
    FOREIGN KEY (tenant_id, post_id) REFERENCES posts(tenant_id, id),
    FOREIGN KEY (tenant_id, image_id) REFERENCES images(tenant_id, id)
);
