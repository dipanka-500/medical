# =====================================================================
#  MEDISCAN AI — Makefile
#  Common commands for development and production workflows
# =====================================================================

COMPOSE := docker compose
COMPOSE_PROD := $(COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml
N ?= 5

.PHONY: dev prod build logs health migrate scale-platform backup-db clean

## Start all services in development mode (foreground)
dev:
	$(COMPOSE) up

## Start all services in production mode (detached)
prod:
	$(COMPOSE_PROD) up -d

## Build all container images
build:
	$(COMPOSE) build

## Tail logs from all services
logs:
	$(COMPOSE) logs -f --tail=100

## Check the platform health endpoint
health:
	@curl -fsSL http://localhost:$${PLATFORM_PORT:-8000}/api/v1/health/live && echo ""

## Run Alembic database migrations
migrate:
	$(COMPOSE) exec platform alembic upgrade head

## Scale the platform service to N replicas (default: 5)
## Usage: make scale-platform N=5
scale-platform:
	$(COMPOSE_PROD) up -d --scale platform=$(N)

## Dump the PostgreSQL database into the backup volume
backup-db:
	$(COMPOSE) exec postgres sh -c 'pg_dump -U $${POSTGRES_USER:-medai} $${POSTGRES_DB:-medai} | gzip > /backups/medai_$$(date +%Y%m%d_%H%M%S).sql.gz'
	@echo "Backup saved to postgres_backup volume"

## Tear down all services and remove volumes
clean:
	$(COMPOSE) down -v
