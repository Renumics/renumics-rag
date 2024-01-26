#!/usr/bin/env python3
"""
Explore embeddings database.
"""

from pathlib import Path

import pandas as pd
import typer
from renumics import spotlight
from typing_extensions import Annotated
from langchain.vectorstores.chroma import Chroma

from assistant import get_embeddings_model


app = typer.Typer()


@app.command()
def answer_question(
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
    visualize: Annotated[bool, typer.Option(help="Visualize created tables.")] = True,
) -> None:
    """
    Load sources and chat history from a Polytec Assistant settings file, export
    and optionally visalize them.
    """

    embeddings_model = get_embeddings_model(embeddings_model_name)

    # get docs from db
    docs_vector_store = Chroma(
        collection_name=db_collection,
        embedding_function=embeddings_model,
        persist_directory=str(db_directory),
    )
    response = docs_vector_store.get(include=["metadatas", "documents", "embeddings"])
    docs_df = pd.DataFrame(
        {
            "id": response["ids"],
            "source": [metadata.get("source") for metadata in response["metadatas"]],
            "page": [metadata.get("page", -1) for metadata in response["metadatas"]],
            "document": response["documents"],
            "embedding": response["embeddings"],
        }
    )

    # get questions from db
    questions_vector_store = Chroma(
        collection_name=f"questions_{db_collection}",
        embedding_function=embeddings_model,
        persist_directory="./db-out",
    )
    response = questions_vector_store.get(
        include=["metadatas", "documents", "embeddings"]
    )
    questions_df = pd.DataFrame(
        {
            "id": response["ids"],
            "question": response["documents"],
            "embedding": response["embeddings"],
            "answer": [metadata.get("answer") for metadata in response["metadatas"]],
            "sources": [
                metadata.get("sources").split(",") for metadata in response["metadatas"]
            ],
        }
    )

    docs_df["used_by_n_questions"] = docs_df["id"].apply(
        lambda id: questions_df["sources"].apply(lambda sources: id in sources).sum()
    )
    docs_df["used_by"] = [
        q["id"] if src_id in q["sources"] else None
        for i, q in questions_df.iterrows()
        for src_id in docs_df["id"]
    ]
    questions_df["used_by"] = questions_df["id"]
    if visualize:
        df = pd.concat([docs_df, questions_df], ignore_index=True)
        print(df)
        spotlight.show(df)


if __name__ == "__main__":
    app()
