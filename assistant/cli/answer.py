#!/usr/bin/env python3
from pathlib import Path
from typing import List

import typer
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from typing_extensions import Annotated

from assistant import get_chat_model, get_embeddings_model

app = typer.Typer()


def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents as following:

    Content: ...
    Source: ...
    ...
    """
    return "\n\n".join(
        f"Content: {doc.page_content}\nSource: {doc.metadata.get('source', f'Document {i + 1}')}"
        for i, doc in enumerate(docs)
    )


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
    ] = Path("./db"),
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
    embeddings_model = get_embeddings_model(embeddings_model_name)
    chat_model = get_chat_model(chat_model_name)

    vectorstore = Chroma(
        collection_name=db_collection,
        embedding_function=embeddings_model,
        persist_directory=str(db_directory),
    )
    retriever = vectorstore.as_retriever()

    template = """You are an assistant for question-answering tasks on Formula One (F1) documentation.
Given the following extracted parts of a long document and a question, create a final answer with used references (named "sources").
Keep the answer concise. If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return used sources in your answer.

QUESTION: {question}
=========
{source_documents}
=========
FINAL ANSWER: """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            source_documents=(lambda x: format_docs(x["source_documents"]))
        )
        | prompt
        | chat_model
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {  # type: ignore
            "source_documents": retriever,
            "question": RunnablePassthrough(),
        }
    ).assign(answer=rag_chain_from_docs)

    for question in questions:
        rag_answer = rag_chain_with_source.invoke(question)
        print(f"\nQuestion: {rag_answer['question']}")
        print(f"Answer: {rag_answer['answer']}")
        print("Sources:")
        for doc in rag_answer["source_documents"]:
            print(f"\nContent: {doc.page_content}")
            print(
                "Metadata: "
                + ", ".join(f"{key}: {value}" for key, value in doc.metadata.items())
            )


if __name__ == "__main__":
    app()
