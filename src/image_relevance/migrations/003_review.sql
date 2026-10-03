ALTER TABLE suggestions ADD COLUMN decision TEXT
    CHECK (decision IN ('approved', 'rejected'));
ALTER TABLE suggestions ADD COLUMN note TEXT NOT NULL DEFAULT '';
ALTER TABLE suggestions ADD COLUMN reviewed_at TEXT;
