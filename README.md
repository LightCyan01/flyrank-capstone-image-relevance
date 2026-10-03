# Image Relevance

Image Relevance uses YOLO to classify animal photos. BLIP generates image captions, and MiniLM matches those captions to articles. All three models run locally. A subject check rejects mismatches such as a wolf photo for a red-fox article. Suggestions include scores and explanations, with an API for approving or rejecting each pairing.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), open a terminal in this repository, and run:

```sh
uv run image-relevance demo
```

This installs the locked Python 3.12 dependencies, creates `.env` with a random token, downloads the demo photos and models, processes the library, and starts the server. The first run needs internet access and takes longer than later launches. Inference runs locally on the CPU.

Open [Swagger](http://127.0.0.1:8000/docs). Click Authorize and paste only the token after `"demo":` in `.env`, without quotes or the word `Bearer`. Keep `.env` private.

Use the three numbered endpoints under "Match an article":

1. Open `POST /matches`, click Try it out, and Execute. The example fox article is filled in. Leave `files` empty to search the demo library. For your own images, click Add string item, then Choose File for each image, and edit the article fields.
2. Open `GET /matches/latest`, click Try it out, and Execute. Repeat while the status is `queued` or `running`. The result shows up to five suggestions with filenames, captions, scores and an `image_number`. `no_confident_match` includes reasons.
3. Open `POST /matches/{match_id}/review` and click Try it out. Copy `match_id` and the chosen `image_number` from step 2, set `decision` to `approved` or `rejected`, and Execute. Approval cannot override the mismatch guard.

Submitting the article starts processing automatically. Repeating the same submission reuses the match. If processing fails, inspect its errors and `/alerts`, fix the cause, and submit it again to retry failed items.

For separate startup and seeding, use these commands. Seeding works before startup or in a second terminal while the server runs:

```sh
uv run image-relevance serve
uv run image-relevance seed
```

The demo contains 40 Wikimedia Commons photos across five animal groups, one generated blank image, and eighteen posts. Downloads are checked against the committed checksums. Cached image descriptions and embeddings are reused.

## YOLO models

Put classification weights in `models/`. The default, `models/yolo26s-cls.pt`, downloads automatically. To use a custom model, put its file there and change `.env`:

```dotenv
IMAGE_RELEVANCE_VISION_MODEL=my-animal-model.pt
```

An absolute path also works. Restart the server and submit the article again after changing models. Give updated weights a new filename so the worker refreshes cached metadata. BLIP and MiniLM download automatically into `data/model-cache/`.

Use classification weights (`*-cls.pt`) with descriptive class names such as `red fox` or `gray wolf`. Detection weights such as `yolo26s.pt` are not supported by this pipeline.

## Other endpoints

All resource endpoints require `Authorization: Bearer <your local token>`. The Process images, Library, Advanced and Diagnostics sections in Swagger expose the individual operations:

| Action | Endpoint |
| --- | --- |
| Upload a JPEG, PNG or WebP, maximum 10 MiB / 25 million pixels | `POST /images` (`file` multipart field) |
| Create an article and start processing | `POST /posts` (`title`, `content`) |
| Enqueue images and/or posts | `POST /jobs` (`image_ids`, `post_ids`; `Idempotency-Key` header) |
| Inspect batch progress and errors | `GET /jobs/{job_id}` |
| Inspect classified images and posts | `GET /images`, `GET /posts` |
| View an uploaded image in Swagger | `GET /images/{image_id}/file` |
| Rank and explain candidates | `GET /posts/{post_id}/images` |
| Force one candidate through the guard | `GET /posts/{post_id}/images/{image_id}` |
| Inspect a pairing | `GET /suggestions/{suggestion_id}` |
| Approve or reject | `POST /suggestions/{suggestion_id}/review` (`decision`, optional `note`) |
| Inspect costs and permanent failures | `GET /costs`, `GET /alerts` |

Uploads and posts are deduplicated by content. Reusing a batch key with different IDs returns 409. A single background worker processes the SQLite queue, retries transient failures up to three times, and resumes unfinished jobs after a restart. After a failed batch, inspect `/alerts`, fix the cause, and submit a new batch key.

Bearer tokens map to tenants. Each tenant has its own records and a default limit of 500 model calls per UTC day. The call ledger records attempts, durations and outcomes at $0 API spend; it does not measure hardware or electricity costs. Uploaded files, the database and model caches are stored locally and excluded from Git.

## Matching and limits

```text
Images -> background queue -> YOLO + BLIP -> validated metadata -> MiniLM
Posts  -> background queue -------------------------------------> MiniLM
                                      |
                                SQLite storage
                                      |
                     similarity ranking -> mismatch guard -> review API
```

YOLO supplies the animal label and confidence. BLIP describes the image, prompted with that label; attributes are selected from details mentioned in its caption. Pydantic validates the combined metadata before storage. MiniLM produces normalized 384-dimensional embeddings for captions and articles. Matching ranks images by cosine similarity, then checks the subject and the default thresholds: 0.55 vision confidence and 0.40 similarity. Images with low confidence or unsupported labels are marked `needs_review`.

The subject check covers fox, wolf, dog, bear and deer, including common scientific names. An article about red foxes rejects other fox species. Articles with unknown or mixed subjects return `no_confident_match`. Review approval cannot override a failed subject or confidence check.

This is an animal-matching demo with one server process and a scan over stored vectors. BLIP can miss details or invent them; YOLO confidence does not measure caption accuracy. The default ImageNet classifier has no deer class, so deer support needs custom classification weights. Confident classification errors are still possible. Articles about unsupported subjects require manual selection, and the guard does not understand arbitrary negation or comparisons.

The app uses SQLite in place of the brief's PostgreSQL setup and runs YOLO, BLIP and MiniLM directly in Python instead of Ollama. Images and article text stay on the machine running the server. No hosted AI account or model API key is needed.

## Evaluation and results

```sh
uv run image-relevance seed
uv run image-relevance eval
```

Evaluation saves every labeled post's expected and predicted image in `data/evaluation.json`. It ranks only the seeded photos and ignores human review decisions when measuring model accuracy. An abstention or a different photo of the same animal counts as incorrect. `corpus/posts.json` contains twelve evaluation posts, five separate calibration examples and one unsupported-subject probe. The calibration examples are scored before the evaluation set; thresholds stay at 0.55 confidence and 0.40 similarity.

The default models scored 1/12 (8.33%) for the exact image and 10/12 (83.33%) for the animal subject. Nine posts selected another photo of the correct animal; two deer posts returned no match. Exact-image retrieval is poor on these labels, which often designate one of several suitable animal photos. Of the 41 images, 12 were flagged for review. A fresh seed processed 141 successful model calls at $0 API spend: 41 YOLO calls, 41 BLIP calls and 59 embedding calls. Calibration scored 0/5 exact photos and 4/5 animal subjects; the thresholds were not tuned on the evaluation posts.

For the fox behavior article, the best fox, wolf and dog cosine scores were 0.724214, 0.415262 and 0.234661. The wolf and dog candidates were rejected. A sunflower article also returned no match. These results describe this small demo set; they do not establish general accuracy.

## Inspiration and licenses

This project draws on my earlier work in [Yolo-ToolKit](https://github.com/LightCyan01/Yolo-ToolKit) for YOLO loading and inference, [TrainKit](https://github.com/LightCyan01/TrainKit) for image verification and safe file writes, and [mediabatch](https://github.com/LightCyan01/mediabatch) for sequential RGB image processing.

Photo sources, authors, licenses and checksums are listed in `corpus/images.json`. The blank image is generated locally. Application code is MIT licensed; [Ultralytics/model licensing](https://www.ultralytics.com/license) applies separately to its software and weights.
