SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

.PHONY: help
help: ## Print this help message
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)"

.PHONY: init
init: ## Locally install all dev dependencies
	poetry install --all-extras

.PHONY: init-cpu
init-cpu: ## Locally install all dev dependencies with CPU support
init-cpu: init
	poetry run pip install torch torchvision sentence-transformers accelerate \
		--extra-index-url https://download.pytorch.org/whl/cpu

.PHONY: init-gpu
init-gpu: ## Locally install all dev dependencies with GPU support
init-gpu: init
	poetry run pip install torch torchvision sentence-transformers accelerate

.PHONY: clean
clean: ## Clean project
	rm -rf .ruff_cache/ .mypy_cache/

.PHONY: check-format
check-format: ## Check code formatting
	poetry run black --check .

.PHONY: format
format: ## Fix code formatting
	poetry run black .

.PHONY: typecheck
typecheck: ## Typecheck all source files
	poetry run mypy -p assistant

.PHONY: lint
lint: ## Lint all source files
	poetry run ruff assistant

.PHONY: run
run: ## Run web app
	poetry run streamlit run assistant/app.py

.PHONY: build-wheel
build-wheel: ## Build package
	poetry build -f wheel

.PHONY: build-image
build-image: ## Build docker image
	docker build -t renumics-rag -f Dockerfile .

.PHONY: run-image-openai
run-image-openai: ## Build docker image
run-image-openai: build-image
	docker run -it --rm -e OPENAI_API_KEY=$$OPENAI_API_KEY -p 8000:8000 renumics-rag

.PHONY: run-image-azure
run-image-azure: ## Build docker image
run-image-azure: build-image
	docker run -it --rm \
		-e OPENAI_API_TYPE=$$OPENAI_API_TYPE \
		-e OPENAI_API_VERSION=$$OPENAI_API_VERSION \
		-e AZURE_OPENAI_API_KEY=$$AZURE_OPENAI_API_KEY \
		-e AZURE_OPENAI_ENDPOINT=$$AZURE_OPENAI_ENDPOINT -p 8000:8000 renumics-rag

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
