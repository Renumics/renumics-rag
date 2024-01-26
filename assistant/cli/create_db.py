#!/usr/bin/env python3
from pathlib import Path

import typer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import BSHTMLLoader, DirectoryLoader
from typing_extensions import Annotated

from assistant import get_chromadb, get_embeddings_model, stable_hash

app = typer.Typer()


@app.command()
def create_db(
    docs_directory: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=False, help="Directory with documents to index."
        ),
    ] = Path("./data/docs"),
    embeddings_model_name: Annotated[
        str, typer.Option("-e", "--embeddings", help="Name of embeddings model.")
    ] = "WhereIsAI/UAE-Large-V1",
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
    Index documents into database.
    """
    embeddings_model = get_embeddings_model(embeddings_model_name, model_type="hf")

    loader = DirectoryLoader(
        str(docs_directory),
        glob="*.html",
        loader_cls=BSHTMLLoader,
        loader_kwargs={"open_encoding": "utf-8"},
        recursive=True,
        show_progress=True,
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    splits = text_splitter.split_documents(docs)

    vectorstore = get_chromadb(db_directory, embeddings_model, db_collection)

    doc_ids = list(map(stable_hash, splits))

    vectorstore.add_documents(splits, ids=doc_ids)
    vectorstore.persist()


if __name__ == "__main__":
    app()
