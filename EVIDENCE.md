## Seed and evaluation

Run from the repository root:

```sh
uv run image-relevance seed
uv run image-relevance eval
```

The first command created a new database and processed the complete corpus. Model download and progress lines are omitted here:

```text
Processed 59/59 items
{"seeded_images": 41, "seeded_posts": 18, "status": "completed", "calls": 141, "cost_usd": 0.0}
```

The evaluation output, also saved in `data/evaluation.json`, includes:

```json
{
  "posts": 12,
  "correct": 1,
  "top1_precision": 0.08333333333333333,
  "subject_precision": 0.8333333333333334,
  "vision_model": "yolo26s-cls.pt",
  "caption_model": "Salesforce/blip-image-captioning-base",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "min_confidence": 0.55,
  "min_similarity": 0.4
}
```

| Post | Labeled image | First allowed image |
| --- | --- | --- |
| fox-behavior | fox-01 | fox-07 |
| fox-taxonomy | fox-01 | fox-07 |
| fox-wild | fox-01 | fox-07 |
| wolf-pack | wolf-01 | wolf-02 |
| wolf-taxonomy | wolf-01 | wolf-04 |
| wolf-winter | wolf-01 | wolf-01 |
| dog-family | dog-01 | dog-03 |
| dog-care | dog-01 | dog-03 |
| bear-brown | bear-01 | bear-03 |
| bear-grizzly | bear-01 | bear-04 |
| deer-elk | deer-01 | No match |
| deer-herd | deer-01 | No match |

The five separate calibration posts scored 0/5 exact photos and 4/5 animal subjects. Thresholds and labels were retained. Evaluation considers only the seeded images and counts abstentions as incorrect. It measures model output before human review decisions; rejecting a pairing or uploading more photos does not alter the benchmark corpus.

## Backend verification

The verification runner reported:

```text
PASS GATES:G1: review and guided workflow validate input, isolate tenants, preserve decisions and refuse unsafe or stale approvals
PASS GATES:G2: public seed and evaluation run actual local models across the corpus and measure labeled top-1 precision
PASS GATES:G3: pipeline, matching and design regression checks pass
PASS GATES:G4: source and local verification scripts pass lint and type checks
```

The model check also printed this measured summary:

```json
{
  "images": 41,
  "posts": 18,
  "flagged": 12,
  "calls": 141,
  "cost_usd": 0.0,
  "exact_correct": 1,
  "evaluated_posts": 12,
  "subject_precision": 0.8333333333333334,
  "fox_first": "fox-07.png",
  "scientific_first": "fox-07.png",
  "wolf_refusal": [
    "Animal category mismatch: expected fox, detected white wolf",
    "Similarity below threshold: 0.305 < 0.400"
  ],
  "no_match": "no_confident_match"
}
```

## Section 6 proofs

Each row names the check and its observed output. The G1 through G4 lines above are the runner's aggregate results. The public API exposes the corresponding records for inspection after seeding.

| Requirement | Check and proof |
| --- | --- |
| Schema-valid vision output; invalid output refused | Local `check_worker` injects invalid metadata and captions and asserts that neither is saved: `PASS GATES:G3`. Real `check_models` validates all 41 stored metadata records: `PASS GATES:G2`. |
| Low confidence flagged | `check_models` verifies every stored status against confidence and category: `"flagged": 12`. The blank image returned `needs_review`, subject `dishwasher`, category `other`, confidence `0.09201736748218536`. |
| Background batches and retries | `check_worker` verifies a transient failure succeeds on attempt 2; `check_workflow` holds inference while HTTP returns 202: `PASS GATES:G3` and `PASS GATES:G1`. |
| Vision and embedding costs per call | `check_models` verifies successful, attributed calls: 41 classification, 41 caption and 59 embedding calls, `"calls": 141, "cost_usd": 0.0`. `GET /costs?limit=1000` exposes each attempt. |
| Stored embeddings and ranked suggestions | `check_models` validates 41 image vectors and 18 post vectors as normalized arrays of 384 finite values: `PASS GATES:G2`. The browser result below ranks fox-07 first at `0.724214`. |
| Equivalent concepts match | `check_models`: `"fox_first": "fox-07.png", "scientific_first": "fox-07.png"` for the common-name and Vulpes vulpes articles. |
| Wolf mismatch refused | Forced wolf-01 probe: `Animal category mismatch: expected fox, detected white wolf`. The result is `no_confident_match`. |
| Human-readable rejection reasons | The forced probe also reports `Similarity below threshold: 0.305 < 0.400`. `GET /posts/{post_id}/images/{image_id}` returns the candidate, score and explanation. |
| No confident match with reasons | Sunflower probe: `"no_match": "no_confident_match"`; reason `Article has no single supported animal subject; manual selection required`. |
| Persistent data and indexes | `check_migration` preserves earlier jobs/calls/alerts: `PASS GATES:G3`. Live SQLite inspection found four applied migrations and `Foreign keys: []`. Indexes include `images_tenant_status`, `tags_lookup`, `posts_tenant_status`, `items_queue`, `calls_budget`, plus unique indexes for tenant/resource and suggestion pairs. Review fields persist in `suggestions`. |
| Validated approve/reject/inspect API | `check_review` checks 401 without authentication, 404 across tenants, 422 for invalid decisions, 409 for unsafe approval, persistent rejection and exclusion from suggestions: `PASS GATES:G1`. Browser approval below returned 200. |
| Labeled top-1 evaluation | `uv run image-relevance eval`: `"posts": 12, "correct": 1, "top1_precision": 0.08333333333333333`. README reports 8.33%. |
| Architecture and required files | README includes run/seed commands, an ASCII architecture diagram, results and limitations. The submission also includes capstone.yaml, EVIDENCE.md, BUILDLOG.md, .env.example and LICENSE. |

## Swagger workflow

The documented workflow was exercised in the browser on port 8001 with a temporary local token:

```text
POST /matches, default fox article, files left empty
202 {"match_id":"3b41f0c01cff4494b70cf1145fb95b83","status":"queued"}

GET /matches/latest
200: status=suggested, processed=42, total=42
First suggestion: fox-07.png, similarity=0.724214, image_number=22

POST /matches/3b41f0c01cff4494b70cf1145fb95b83/review
{"decision":"approved","note":"Verified in Swagger","image_number":22}
200: decision=approved, note=Verified in Swagger, reviewed_at=2026-10-03 01:36:47
```

IDs and image numbers vary between fresh databases. The automated workflow check separately verified selected file uploads, repeated submissions, use of the existing library and reopening an earlier match.

## Limits of these results

Nine evaluation posts selected another photo of the correct animal; two deer posts returned no match. The default ImageNet classifier has no deer class. BLIP described fox-01 as `a red fox in the grass greeting card`, illustrating that a valid caption can contain invented details. Confidence refers to YOLO classification, not caption accuracy. These results establish behavior on this small corpus and do not establish general retrieval accuracy.

The ledger records $0 model API charges, not electricity or hardware costs. The application uses one local worker and scans stored vectors. Phase commits document a rebuild from an existing prototype; they do not recreate the prototype's original development history.
