#!/usr/bin/env python3
from pathlib import Path
from typing import List

import typer
from typing_extensions import Annotated

from assistant import get_chromadb, get_embeddings_model, get_retriever

app = typer.Typer()


@app.command()
def retrieve(
    questions: Annotated[List[str], typer.Argument(help="Question(s) to answer.")],
    embeddings_model_name: Annotated[
        str, typer.Option("-e", "--embeddings", help="Name of embeddings model.")
    ] = "text-embedding-ada-002",
    db_directory: Annotated[
        Path, typer.Option("--db", help="Directory to persist database in.")
    ] = Path("./db-docs"),
    db_collection: Annotated[
        str,
        typer.Option(
            "--collection", help="Name of database collection to store documents."
        ),
    ] = "docs_store",
) -> None:
    """
    Retrieve documents relevant to question(s) using indexed database.
    """
    embeddings_model = get_embeddings_model(embeddings_model_name)
    vectorstore = get_chromadb(
        persist_directory=db_directory,
        embeddings_model=embeddings_model,
        collection_name=db_collection,
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
