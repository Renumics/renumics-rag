#!/usr/bin/env python3
from pathlib import Path

import typer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import BSHTMLLoader, DirectoryLoader
from typing_extensions import Annotated

from assistant.llm import get_embeddings_model

app = typer.Typer()


@app.command()
def create_db(
    docs_directory: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Directory with document to index.",
        ),
    ] = Path("./data/docs"),
    embeddings_model_name: Annotated[
        str, typer.Option("-m", "--model", help="Name of embeddings model.")
    ] = "text-embedding-ada-002",
    db_directory: Annotated[
        Path,
        typer.Option("-o", "--output", help="Directory to persist database in."),
    ] = Path("./db"),
    db_collection: Annotated[
        str,
        typer.Option(
            "-c", "--collection", help="Name of database collection to store documents."
        ),
    ] = "docs_store",
) -> None:
    embeddings_model = get_embeddings_model(embeddings_model_name)

    loader = DirectoryLoader(
        str(docs_directory),
        glob="**/*.html",
        loader_cls=BSHTMLLoader,
        show_progress=True,
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings_model,
        ids=None,
        collection_name=db_collection,
        persist_directory=str(db_directory),
    )
    vectorstore.persist()


if __name__ == "__main__":
    app()
