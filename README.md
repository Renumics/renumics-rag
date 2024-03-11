# 🤖 RAG Demo

Retrieval-augmented generation assistant demo using [LangChain](https://github.com/langchain-ai/langchain) and [Streamlit](https://github.com/streamlit/streamlit).

## 🛠️ Installation

Setup a virtual environment in the project directory:

```shell
python3.8 -m venv .venv
source .venv/bin/activate  # Linux/MacOS
# .\.venv\Scripts\activate.bat  # Windows CMD
# .\.venv\Scripts\activate.ps1  # PowerShell
pip install -IU pip setuptools wheel
```

Install the RAG demo package and some extra dependencies:

```shell
# For GPU support
pip install git+https://github.com/Renumics/rag-demo.git[all] torch torchvision sentence-transformers accelerate
# For CPU support
# pip install git+https://github.com/Renumics/rag-demo.git[all] torch torchvision sentence-transformers accelerate --extra-index-url https://download.pytorch.org/whl/cpu
```

## ⚒️ Local Setup

If you intend to edit, not simply use, this project, clone the entire repository:

```shell
git clone git@github.com:Renumics/rag-demo.git
```

Then install it in editable mode.

### Via `pip`

Setup virtual environment in the project folder:

```shell
python3.8 -m venv .venv
source .venv/bin/activate  # Linux/MacOS
# .\.venv\Scripts\activate.bat  # Windows CMD
# .\.venv\Scripts\activate.ps1  # PowerShell
pip install -IU pip setuptools wheel
```

Install the RAG demo package and some extra dependencies:

```shell
pip install -e .[all]
# For GPU support
pip install torch torchvision sentence-transformers accelerate
# For CPU support
# pip install torch torchvision sentence-transformers accelerate --extra-index-url https://download.pytorch.org/whl/cpu
```

### Via `poetry`

Install the RAG demo and some extra dependencies:

```shell
poetry install --all-extras
# Torch with GPU support
pip install torch torchvision sentence-transformers accelerate
# Torch with CPU support
# pip install torch torchvision sentence-transformers accelerate --extra-index-url https://download.pytorch.org/whl/cpu
```

Activate the environment (otherwise, prexis all subsequent commands with `poetry run`):

```shell
poetry shell
```

> Note: If you have [Direnv](https://direnv.net/) installed, you can avoid prefixing python commands with `poetry run` by executing `direnv allow` in the project directory. It will activate environment each time you enter the project directory.

### ⚙️ Configuration

If you plan to use OpenAI models, create `.env` with the following content:

```bash
OPENAI_API_KEY="Your OpenAI API key"
```

If you plan to use OpenAI models via Azure, create `.env` with the following content:

```shell
OPENAI_API_TYPE="azure"
OPENAI_API_VERSION="2023-08-01-preview"
AZURE_OPENAI_API_KEY="Your Azure OpenAI API key"
AZURE_OPENAI_ENDPOINT="Your Azure OpenAI endpoint"
```

If you are using Hugging Face models, a `.env` file is not necessary.

Modify parameters if desired in the [settings file](./settings.yaml).

> Note: you can create different settings files and toggle between them by setting the `RAG_SETTINGS` environment variable.

## 🚀 Usage

Create a new `data/docs` directory within the project and place your documents in there (recursive directories are supported).

> Note: at the moment, only HTML files can be indexed but it can be adjusted in the [create-db](assistant/cli/create_db.py) script.

Begin the process by indexing your documents. Execute the following command:

```shell
create-db
```

This will create a `db-docs` directory within the project consisting of indexed documents. To index additional documents, use the `--exist-ok` and `--on-match` flags (refer to `create-db --help` for more information).

Now, you can leverage the indexed documents to answer questions.

To only retrieve relevant documents:

```shell
retrieve "Your question here"
# QUESTION: ...
# SOURCES: ...
```

To answer a question based on the indexed documents:

```shell
answer "Your question here"
# QUESTION: ...
# ANSWER: ...
# SOURCES: ...
```

To start a web application:

```shell
app
```

See `app --help` for available application options.

After submitting some questions, you can explore them using [Renumics Spotlight](https://github.com/Renumics/spotlight):

```shell
explore
```
