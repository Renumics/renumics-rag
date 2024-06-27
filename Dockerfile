FROM python:3.11

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8000

WORKDIR /app

RUN pip install -IU pip setuptools wheel && pip install poetry==1.8.3

COPY pyproject.toml poetry.lock README.md ./
COPY ./assistant/__init__.py ./assistant/

RUN poetry install --extras=openai --sync --without=dev

COPY ./assistant/ ./assistant/
COPY settings.yaml ./

COPY ./db-docs/ ./db-docs/

CMD ["poetry", "run", "streamlit", "run", "assistant/app.py"]
