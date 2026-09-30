# Image Relevance design

This rebuild reuses the existing local prototype. Phase 1 records the design and gathers the image corpus; the pipeline and matching features follow in later phases.

## Problem and scope

Suggest animal photos for articles and explain why unsuitable images were refused. Start with 40 licensed photos in five groups: fox, wolf, dog, bear and deer. Use YOLO for classification, BLIP for captions and MiniLM for text embeddings. All models run locally; SQLite stores the results. Model training and a general-purpose image search platform are outside the scope.

## Image metadata

```json
{
  "subject": "red fox",
  "category": "animal",
  "attributes": ["orange fur", "forest"],
  "caption": "A red fox standing in a forest",
  "confidence": 0.94
}
```

Pydantic will validate model output before storage. Text fields must be nonempty, attributes must be short strings, and confidence must be finite and between zero and one. Invalid output fails processing. Low-confidence or unsupported classifications remain visible with a `needs_review` status.

## Matching rules

Embed captions and article text as normalized 384-dimensional vectors and rank them by cosine similarity. Expand common scientific names such as Vulpes vulpes to red fox. Require one supported article subject and an agreeing image subject. An explicit red-fox article must refuse other fox species. Begin with confidence 0.55 and similarity 0.40; check these thresholds on separate calibration examples before evaluating labeled posts. Return `no_confident_match` with reasons when no image qualifies. Review approval must rerun the guard.

## Data model

| Records | Stored data and constraints |
| --- | --- |
| Tenants | Tenant ID; API tokens stay in the environment. |
| Images and tags | Filename, content checksum, local path, metadata, model names, status and caption vector. Unique checksum per tenant; indexed tags and status. |
| Posts | Title, text, checksum, status and text vector. Unique checksum per tenant. |
| Jobs and items | Request key, resources, progress, attempts, next retry time and errors. Unique request key per tenant; queue indexes. |
| Model calls and alerts | Tenant, job, item, model, outcome, duration and API cost; daily call-count index and permanent-failure alerts. |
| Suggestions | Post/image pair, score, guard reasons, review decision and note. Unique pair per tenant. |

SQL migrations will create the tables. Composite foreign keys and tenant filters will prevent cross-tenant references. Store vectors as JSON arrays and scan them for this small corpus. Upload retries reuse content checksums; batch retries reuse a request key. Record each model attempt before execution and enforce a per-tenant daily limit, even with $0 API spend.

## Layers and API

```text
HTTP validation -> service operations -> SQLite
                         |
                  persistent job queue
                         |
                  YOLO + BLIP -> metadata
                         |
                 MiniLM -> ranking -> guard -> review
```

FastAPI will handle authentication and input validation. Service functions will handle ingestion and matching; the store will own SQL transactions. One worker will run model calls outside HTTP requests, retry transient failures and resume interrupted work.

Phase 2 adds image upload, batch processing, progress, metadata, costs and alerts. Phase 3 adds posts, embeddings, ranking and forced-candidate guard checks. Phase 4 adds the submit/check/review workflow in Swagger, the labeled evaluation and final documentation. No frontend is planned.

The reviewed README stays in the completed project until Phase 4. This repository will use ordinary Python `#` comments only where the reason for a decision needs explaining.
