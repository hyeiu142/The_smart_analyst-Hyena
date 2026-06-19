````md
# AGENTS.md

## Project Overview

Hyena is an Enterprise Multimodal RAG project for financial documents.

Main folders:

- `backend/`: FastAPI backend, RAG, ingestion, retrieval, generation
- `frontend/`: static frontend UI
- `research/`: experimental scripts and debugging code
- `qdrant/`: vector database setup
- `Docs/`: sample documents
- `uploads/`: uploaded runtime files

Current focus:

- Debug and improve code inside `research/image_extraction/`
- Extract charts, tables, and figures from PDF pages
- Save test outputs into `research/image_extraction/test_outputs/`

---

## Agent Rules

When working on this project:

1. Do not rewrite the whole project.
2. Only edit files related to my request.
3. Do not refactor unrelated code.
4. Do not delete files unless I explicitly ask.
5. Do not touch `.env`, secrets, API keys, or credentials.
6. Do not add new dependencies unless necessary and approved.
7. Before editing, inspect the relevant files first.
8. Keep changes small and focused.
9. Preserve the existing folder structure.
10. After editing, explain:
   - which files changed
   - what changed
   - how to test

---

## Debugging Rules

Do not guess from symptoms alone.

If there is an error, first ask me to paste the smallest useful output, such as:

- terminal output
- Python traceback
- Docker logs
- API response
- browser console error
- browser network error

Default debugging process:

1. Identify the likely area:
   - frontend
   - backend
   - worker
   - Redis
   - Qdrant
   - Docker
   - LlamaParse
   - OpenAI
   - research script

2. Ask me to run the relevant command.

3. Wait for me to paste the output.

4. Analyze the output.

5. Suggest the next command or code change.

Do not claim something is fixed without a test result or validation log.

---

## Code Style

For Python code:

- Use simple and readable code.
- Prefer `pathlib.Path` for file paths.
- Avoid hardcoded absolute paths.
- Use type hints for new functions when useful.
- Add comments only when the logic is not obvious.
- Keep scripts runnable from the project root.

Example:

```python
from pathlib import Path

def load_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")
````

---

## Image Extraction Notes

Important folder:

```text
research/image_extraction/
```

Important output folder:

```text
research/image_extraction/test_outputs/
```

Expected output filenames:

```text
page_<page_number>_<region_number>_chart.png
page_<page_number>_<region_number>_table.png
page_<page_number>_<region_number>_figure.png
```

Example:

```text
page_4_01_chart.png
page_4_02_chart.png
page_4_03_table.png
```

When improving image extraction:

1. Inspect the relevant files first.
2. Only modify files inside `research/image_extraction/` unless I ask otherwise.
3. Do not modify backend or frontend.
4. Do not delete existing test outputs.
5. Keep output filenames stable.
6. Explain how to rerun the script.

Relevant files may include:

```text
research/image_extraction/render_pages.py
research/image_extraction/extract_llama_bboxes.py
research/image_extraction/crop_from_llama_bboxes.py
```

Things to check:

* Did the page render correctly?
* Are bounding boxes correct?
* Is the crop missing title, legend, axis labels, or caption?
* Is the crop including too much white space?
* Are tiny or invalid images skipped correctly?

---

## Common Commands

Start all Docker services:

```bash
make up
```

Stop all Docker services:

```bash
make down
```

Show logs:

```bash
make logs
```

Check running containers:

```bash
docker compose ps
```

Backend health check:

```bash
curl http://localhost:8001/api/v1/health/
```

Qdrant health check:

```bash
curl http://localhost:8001/api/v1/health/qdrant
```

Redis health check:

```bash
curl http://localhost:8001/api/v1/health/redis
```

Backend logs:

```bash
docker compose logs backend --tail=200
```

Worker logs:

```bash
docker compose logs worker --tail=300
```

---

## Testing Rules

Before saying a change works, validate it with the smallest relevant test.

For backend tests:

```bash
uv run python -m pytest backend/tests/ -v
```

For service health:

```bash
curl http://localhost:8001/api/v1/health/
curl http://localhost:8001/api/v1/health/qdrant
curl http://localhost:8001/api/v1/health/redis
```

For document status:

```bash
curl http://localhost:8001/api/v1/documents/<DOC_ID>/status
```

For query testing:

```bash
curl -X POST "http://localhost:8001/api/v1/query/" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the revenue?","top_k":5}'
```

If tests cannot be run, explain why.

---

## Git Rules

Do not run destructive commands unless I explicitly ask.

Do not run:

```bash
git reset --hard
git checkout -- .
git clean -fd
```

Do not commit unless I explicitly ask.

Before making large changes, check:

```bash
git status
```

---

## Response Style

Be concise and operational.

For debugging, respond with:

1. what area is likely involved
2. exact command I should run
3. what output I should paste back

For code changes, summarize:

1. files changed
2. what changed
3. how to test
4. any remaining issue

Do not make unrelated changes.

```
```
