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
	poetry run pip install renumics-spotlight torch torchvision sentence-transformers accelerate

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
