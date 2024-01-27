#!/usr/bin/env python3
from typing import List

import typer
from langchain_core.documents import Document
from typing_extensions import Annotated

from assistant import (
    get_chromadb,
    get_embeddings_model,
    get_llm,
    get_rag_chain,
    get_retriever,
    parse_model_name,
    stable_hash,
)
from assistant.const import EMBEDDINGS_MODEL_NAME_HELP, LLM_NAME_HELP
from assistant.settings import settings

app = typer.Typer()


@app.command()
def answer(
    questions: Annotated[List[str], typer.Argument(help="Question(s) to answer.")],
    llm_name: Annotated[
        str, typer.Option("--llm", help=LLM_NAME_HELP)
    ] = settings.full_llm_name,
    embeddings_model_name: Annotated[
        str, typer.Option("--embeddings", help=EMBEDDINGS_MODEL_NAME_HELP)
    ] = settings.full_embeddings_model_name,
) -> None:
    """
    Answer question(s) using indexed database.
    """
    embeddings_model = get_embeddings_model(*parse_model_name(embeddings_model_name))
    llm = get_llm(*parse_model_name(llm_name))
    docs_vectorstore = get_chromadb(
        persist_directory=settings.docs_db_directory,
        embeddings_model=embeddings_model,
        collection_name=settings.docs_db_collection,
    )
    retriever = get_retriever(docs_vectorstore)

    rag_chain = get_rag_chain(retriever, llm)

    questions_vectorstore = get_chromadb(
        persist_directory=settings.questions_db_directory,
        embeddings_model=embeddings_model,
        collection_name=settings.questions_db_collection,
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
