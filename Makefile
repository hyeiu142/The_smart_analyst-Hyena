.PHONY: up down logs dev test clean

# run all services
up: 
	docker compose up -d

# stop all services
down: 
	docker compose down

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