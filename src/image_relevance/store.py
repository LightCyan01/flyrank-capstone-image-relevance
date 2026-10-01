import sqlite3
from contextlib import contextmanager
from pathlib import Path

class Store:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "image-relevance.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS migrations (name TEXT PRIMARY KEY)")
            for migration in sorted((Path(__file__).parent / "migrations").glob("*.sql")):
                if not db.execute(
                    "SELECT 1 FROM migrations WHERE name=?", (migration.name,)
                ).fetchone():
                    # A failed migration rolls back the entire schema change.
                    name = migration.name.replace("'", "''")
                    db.executescript(
                        "BEGIN IMMEDIATE;\n"
                        + migration.read_text(encoding="utf-8")
                        + f"\nINSERT INTO migrations VALUES ('{name}');\nCOMMIT;"
                    )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def rows(self, query: str, values: tuple = ()) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute(query, values)]

    def get(self, table: str, tenant: str, resource_id: str) -> dict:
        if table not in {"images", "jobs"}:
            raise ValueError("Unknown resource type")
        rows = self.rows(f"SELECT * FROM {table} WHERE tenant_id=? AND id=?", (tenant, resource_id))
        if not rows:
            raise LookupError("Resource not found")
        return rows[0]

    def tenant(self, tenant: str) -> None:
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO tenants(id) VALUES (?)", (tenant,))
