# RAG Demo

Retrieval-Augmented Generation Assistant Demo

## Installation

Setup virtual environment in the project folder:

```shell
python3.8 -m venv .venv
source .venv/bin/activate  # Linux/MacOS
# .\.venv\Scripts\activate.bat  # Windows CMD
# .\.venv\Scripts\activate.ps1  # PowerShell
pip install -IU pip setuptools wheel
```

Install RAG demo package and some extra dependencies:

```shell
# Torch with GPU support
pip install git+https://github.com/Renumics/rag-demo.git[all] pandas renumics-spotlight torch torchvision sentence-transformers accelerate
# Torch with CPU support
# pip install git+https://github.com/Renumics/rag-demo.git[all] pandas renumics-spotlight torch torchvision sentence-transformers accelerate --extra-index-url https://download.pytorch.org/whl/cpu
```

## Local Setup

If are going not only to use, but also to modify this project, it makes sense to clone the whole project:

```
git clone git@github.com:Renumics/rag-demo.git
```

and install it editable.

### Via `pip`

Setup virtual environment in the project folder:

```shell
python3.8 -m venv .venv
source .venv/bin/activate  # Linux/MacOS
# .\.venv\Scripts\activate.bat  # Windows CMD
# .\.venv\Scripts\activate.ps1  # PowerShell
pip install -IU pip setuptools wheel
```

Install RAG demo package and some extra dependencies:

```shell
pip install -e .[all]
# Torch with GPU support
pip install pandas renumics-spotlight torch torchvision sentence-transformers accelerate
# Torch with CPU support
# pip install pandas renumics-spotlight torch torchvision sentence-transformers accelerate --extra-index-url https://download.pytorch.org/whl/cpu
```

### Via `poetry`

Install RAG demo and some extra dependencies:

```shell
poetry install --all-extras
# Torch with GPU support
pip install pandas renumics-spotlight torch torchvision sentence-transformers accelerate
# Torch with CPU support
# pip install pandas renumics-spotlight torch torchvision sentence-transformers accelerate --extra-index-url https://download.pytorch.org/whl/cpu
```

Activate environment (otherwise, all further commands should be prefixed with `poetry run`):

```shell
poetry shell
```

> Note: If you have [Direnv](https://direnv.net/) installed, you can avoid prefixing python commands with `poetry run` by executing `direnv allow` in the project folder. It will activate environment each time you enter the project folder.

### Settings

If you are going to use OpenAI models, create `.env` with the following content:

```bash
OPENAI_API_KEY="Your OpenAI API key"
```

If you are going to use OpenAI models via Azure, create `.env` with the following content:

```shell
OPENAI_API_TYPE="azure"
OPENAI_API_VERSION="2023-08-01-preview"
AZURE_OPENAI_API_KEY="Your Azure OpenAI API key"
AZURE_OPENAI_ENDPOINT="Your Azure OpenAI endpoint"
```

If you are going to use OpenAI models via Azure, create empty `.env` file.

Navigate to [settings file](./settings.yaml) and adjust parameters if needed.

> Note: you can create multiple settings files and switch between them by setting `RAG_SETTINGS` environment variable.

## Usage

Create folder `data/docs` in the project folder and place your documents in there (recursive folder structure is supported).

> Note: at the moment only HTML files are supported for indexing but it can be adjusted in the [create-db](assistant/cli/create_db.py) script.

First step is to index your documents. To do it, execute the following command:

```shell
create-db
```

It should create `db-docs` folder in the project with indexed documents inside. To index new documents, use `--exist-ok` and `--on-match` flags (run `create-db --help` to see more).

Now you can use the indexed documents to answer your questions.

To only retrieve relevant documents:

```shell
retrieve "Your question here"
# QUESTION: ...
# SOURCES: ...
```

Answer a question using indexed documents:

```shell
answer "Your question here"
# QUESTION: ...
# ANSWER: ...
# SOURCES: ...
```

Start web app:

```shell
app
```

For available app options, see `app --help`.

After you asked some questions you can explore them in [Renumics Spotlight](https://github.com/Renumics/spotlight):

```shell
explore
```
