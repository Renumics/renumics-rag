#!/usr/bin/env python3
from pathlib import Path
from typing import List

import typer
from typing_extensions import Annotated

from langchain_core.documents import Document

from assistant import (
    get_chat_model,
    get_chromadb,
    get_embeddings_model,
    get_rag_chain,
    get_retriever,
    stable_hash,
)

app = typer.Typer()


@app.command()
def answer(
    questions: Annotated[List[str], typer.Argument(help="Question(s) to answer.")],
    chat_model_name: Annotated[
        str, typer.Option("-m", "--model", help="Name of chat model.")
    ] = "gpt-4",
    embeddings_model_name: Annotated[
        str, typer.Option("-e", "--embeddings", help="Name of embeddings model.")
    ] = "text-embedding-ada-002",
    db_directory: Annotated[
        Path, typer.Option("--db", help="Directory to persist database in.")
    ] = Path("./db-docs"),
    db_out_directory: Annotated[
        Path,
        typer.Option("--db-out", help="Directory to persist databse of outputs in."),
    ] = Path("./db-out"),
    db_collection: Annotated[
        str,
        typer.Option(
            "--collection", help="Name of database collection to store documents."
        ),
    ] = "docs_store",
) -> None:
    """
    Answer question(s) using indexed database.
    """
    embeddings_model = get_embeddings_model(embeddings_model_name, model_type="hf")
    chat_model = get_chat_model(chat_model_name)
    docs_vectorstore = get_chromadb(
        persist_directory=db_directory,
        embeddings_model=embeddings_model,
        collection_name=db_collection,
    )
    retriever = get_retriever(docs_vectorstore)

    rag_chain = get_rag_chain(retriever, chat_model)

    questions_vectorstore = get_chromadb(
        persist_directory=db_out_directory,
        embeddings_model=embeddings_model,
        collection_name=f"questions_{db_collection}",
    )

    for question in questions:
        print(f"QUESTION: {question}")
        rag_answer = rag_chain.invoke(question)

        # store question and answer in db
        questions_vectorstore.add_documents(
            [
                Document(
                    page_content=question,
                    metadata={
                        "answer": rag_answer["answer"],
                        "sources": ",".join(
                            map(stable_hash, rag_answer["source_documents"])
                        ),
                    },
                )
            ]
        )

        print(f"ANSWER: {rag_answer['answer']}")
        print("SOURCES:")
        for doc in rag_answer["source_documents"]:
            print(f"CONTENT: {doc.page_content}")
            print(
                "METADATA: "
                + ", ".join(f"{key}: {value}" for key, value in doc.metadata.items())
            )


if __name__ == "__main__":
    app()
