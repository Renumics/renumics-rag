SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

.PHONY: help
help: ## Print this help message
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)"

.PHONY: init
init: ## Locally install all dev dependencies (CPU support)
	uv sync --all-extras --no-extra cu130

.PHONY: init-gpu
init-gpu: ## Locally install all dev dependencies (GPU support)
	uv sync --all-extras --no-extra cpu

.PHONY: clean
clean: ## Clean project
	rm -rf .venv/ .ruff_cache/ .mypy_cache/

.PHONY: check-format
check-format: ## Check code formatting
	uv run black --check .

.PHONY: format
format: ## Fix code formatting
	uv run black .

.PHONY: typecheck
typecheck: ## Typecheck all source files
	uv run mypy -p assistant

.PHONY: lint
lint: ## Lint all source files
	uv run ruff assistant

.PHONY: run
run: ## Run web app
	uv run app

.PHONY: build-wheel
build-wheel: ## Build package
	uv build --wheel

.PHONY: build-image
build-image: ## Build prod docker image
	docker build --target prod -t renumics-rag -f Dockerfile .

.PHONY: run-image-openai
run-image-openai: ## Run prod image with OpenAI credentials
run-image-openai: build-image
	docker run -it --rm -p 8000:8000 --network=host -e OPENAI_API_KEY=$$OPENAI_API_KEY renumics-rag

.PHONY: run-image-azure
run-image-azure: ## Run prod image with Azure OpenAI credentials
run-image-azure: build-image
	docker run -it --rm -p 8000:8000 --network=host \
		-e OPENAI_API_TYPE=$$OPENAI_API_TYPE \
		-e OPENAI_API_VERSION=$$OPENAI_API_VERSION \
		-e AZURE_OPENAI_API_KEY=$$AZURE_OPENAI_API_KEY \
		-e AZURE_OPENAI_ENDPOINT=$$AZURE_OPENAI_ENDPOINT \
		renumics-rag

.PHONY: up
up: ## Start dev environment via Docker Compose
	docker compose up

.PHONY: down
down: ## Stop dev environment via Docker Compose
	docker compose down

.PHONY: docker-login
docker-login: ## Log in to Azure registry
	docker login -u "$$AZURE_REGISTRY_USERNAME" -p "$$AZURE_REGISTRY_PASSWORD" "$$AZURE_REGISTRY"

.PHONY: release-image
release-image: ## Tag and push image to Azure registry
release-image: docker-login build-image
	TIMESTAMP="$(shell date '+%Y-%m-%d_%H-%M-%S')"
	docker tag renumics-rag "$${AZURE_REGISTRY}/renumics-rag:$$TIMESTAMP"
	docker push "$${AZURE_REGISTRY}/renumics-rag:$$TIMESTAMP"
	docker tag renumics-rag "$${AZURE_REGISTRY}/renumics-rag:latest"
	docker push "$${AZURE_REGISTRY}/renumics-rag:latest"
