# Build log

This repository is pretty much a complete rebuild of an earlier prototype that I've been working on for a while now that combines what I've learned from my previous similar projects ([Yolo-ToolKit](https://github.com/LightCyan01/Yolo-ToolKit), [TrainKit](https://github.com/LightCyan01/TrainKit), [mediabatch](https://github.com/LightCyan01/mediabatch)) most are which are AI assisted so I used it as my base.

## Phase 1: Design and corpus

I've let AI assist me and compared what I've worked on and the capstone PDF that way I knew what needed to be added in or removed. The design kept my local YOLO classification, RGB image loading and sequential inference. I chose SQLite and local models (YOLO) in place of PostgreSQL and Ollama.

The first commits added the design, strict metadata schema and a manifest of 40 licensed Wikimedia Commons photos across five animal groups. The manifest records source URLs, credits, licenses and checksums.

## Phase 2: Image understanding

I've let AI assist me in simplifying settings, storage, uploads and persistent queues.

The AI used too much scaffolding and confusing editor/type patterns which I had to change and had it help me simplify instead of trying to make it so complicated. This one uses ordinary functions and explicit dependency annotations. Swagger initially rendered upload controls incorrectly and the upload schema was corrected to show file pickers.

## Phase 3: Matching

I've had AI do most of the working during this part with heavy guidance from me. This added scientific-name aliases, normalized MiniLM embeddings, post processing and similarity ranking and a mismatch guard.

This also included validation where it rejects invalid vectors, stale model output, uncertain candidates and cross-tenant requests. Failed embedding attempts reuse completed vision metadata. Real inference ranked fox first for common and scientific names, refused the wolf, and returned no match for an unrelated article.

## Phase 4: Review and evaluation

AI helped with the review and eval one of which the evaluation loop follows each labeled post: rank the seeded images, take the first allowed candidate, compare its image key with the label, and count the results. 5 separate calibration examples are measured before evaluation and human review decisions and later uploads no longer change the model benchmark. Bankend checks also review persistence and unsafe approvals, tenant isolation, malformed inputs, retries, restart recovery and budgets.
