# RAG Demo

Retrieval-Augmented Generation Assistant Demo

## Development Setup

Enable [Direnv](https://direnv.net/):

```shell
direnv allow
```

Install dependencies via `make init-gpu` for GPU support ar via `make init-cpu` for CPU support (only relevant for local models).

Create `.env` with the following content:

```shell
# For OpenAI usage
OPENAI_API_KEY="Your OpenAI API key"
# OR for Azure OpenAI usage
OPENAI_API_TYPE="azure"
OPENAI_API_VERSION="2023-08-01-preview"
AZURE_OPENAI_API_KEY="Your Azure OpenAI API key"
AZURE_OPENAI_ENDPOINT="Your Azure OpenAI endpoint"
# OR empty for Hugging Face usage
```

Index documents:

```shell
create-db
```

For more options, see `create-db --help`.

Retrieve documents:

```shell
retrieve "When was the first ever F1 race?"
```

For more options, see `retrieve --help`.

Answer a question:

```shell
answer "When was the first ever F1 race?"
```

For more options, see `answer --help`.

Start web app:

```shell
streamlit run assistant/app.py
```

After you asked some questions you can explore them in [Renumics Spotlight](https://github.com/Renumics/spotlight):

```shell
explore
```
