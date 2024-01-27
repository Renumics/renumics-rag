#!/usr/bin/env python3
from pathlib import Path

import typer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import BSHTMLLoader, DirectoryLoader
from typing_extensions import Annotated

from assistant import get_chromadb, get_embeddings_model, parse_model_name, stable_hash
from assistant.const import EMBEDDINGS_MODEL_NAME_HELP
from assistant.settings import settings

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
        str, typer.Option("--embeddings", help=EMBEDDINGS_MODEL_NAME_HELP)
    ] = settings.full_embeddings_model_name,
) -> None:
    """
    Index documents into database.
    """
    embeddings_model = get_embeddings_model(*parse_model_name(embeddings_model_name))

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
    split_ids = list(map(stable_hash, splits))

    docs_vectorstore = get_chromadb(
        settings.docs_db_directory, embeddings_model, settings.docs_db_collection
    )

    docs_vectorstore.add_documents(splits, ids=split_ids)
    docs_vectorstore.persist()


if __name__ == "__main__":
    app()
