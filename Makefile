.PHONY: up down restart logs dev test clean worker dev-all

# run all services
up:
	docker compose up -d

# stop all services
down:
	docker compose down

# restart backend (picks up new .env)
restart:
	docker compose up -d --force-recreate backend

build:
	docker compose build

logs:
	docker compose logs -f

# run backend in dev mode
dev:
	uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

test:
	cd backend && pytest tests/ -v

# run only Qdrant and Redis (dev mode)
infra:
	docker compose up -d qdrant redis

# delete volumes
clean:
	docker compose down -v

# run Celery worker (dev mode - connects to Redis on localhost)
worker:
	REDIS_URL=redis://localhost:6379/0 uv run celery -A backend.app.workers.celery_app.celery_app worker --loglevel=info

# start backend + worker together
dev-all:
	@echo "Starting backend on :8000 and Celery worker..."
	@trap 'kill 0' SIGINT; \
	uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000 & \
	REDIS_URL=redis://localhost:6379/0 uv run celery -A backend.app.workers.celery_app.celery_app worker --loglevel=info & \
	wait

# init Qdrant collections (run once)
init:
	python scripts/init_collections.py
