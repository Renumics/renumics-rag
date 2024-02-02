#!/usr/bin/env python3
from typing import List

import typer
from typing_extensions import Annotated

from assistant import (
    get_chromadb,
    get_embeddings_model,
    get_retriever,
    parse_model_name,
)
from assistant.const import EMBEDDINGS_MODEL_NAME_HELP
from assistant.settings import settings

app = typer.Typer()


@app.command()
def retrieve(
    questions: Annotated[
        List[str], typer.Argument(help="Question(s) to retrieve docs for.")
    ],
    embeddings_model_name: Annotated[
        str, typer.Option("--embeddings", help=EMBEDDINGS_MODEL_NAME_HELP)
    ] = settings.full_embeddings_model_name,
) -> None:
    """
    Retrieve documents relevant to question(s) using indexed database.
    """
    embeddings_model = get_embeddings_model(*parse_model_name(embeddings_model_name))
    vectorstore = get_chromadb(
        embeddings_model, settings.docs_db_directory, settings.docs_db_collection
    )
    retriever = get_retriever(vectorstore)

    for question in questions:
        docs = retriever.get_relevant_documents(question)
        print(f"QUESTION: {question}")
        print("SOURCES:")
        for doc in docs:
            print(f"CONTENT: {doc.page_content}")
            print(
                "METADATA: "
                + ", ".join(f"{key}: {value}" for key, value in doc.metadata.items())
            )


if __name__ == "__main__":
    app()
