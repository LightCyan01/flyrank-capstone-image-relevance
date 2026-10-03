ALTER TABLE tenants ADD COLUMN latest_match_id TEXT REFERENCES jobs(id);
