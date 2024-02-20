from pathlib import Path

import chromadb
import pandas as pd

from assistant import get_chromadb


def _assert_collection_exists(db_directory: Path, db_collection: str) -> None:
    if not db_directory.is_dir():
        raise NotADirectoryError("No vectorstore found at '{db_directory}'.")
    try:
        client_settings = chromadb.config.Settings(
            is_persistent=True, persist_directory=str(db_directory)
        )
        client = chromadb.Client(client_settings)
    except Exception as e:
        raise RuntimeError("Cannot open vectorstore at '{db_directory}'.") from e
    else:
        collection_names = [collection.name for collection in client.list_collections()]
        if db_collection not in collection_names:
            raise RuntimeError(
                f"Collection '{db_collection}' doesn't exists in the "
                f"vectorstore at '{db_directory}'."
            )


def get_docs_df(db_directory: Path, db_collection: str) -> pd.DataFrame:
    try:
        _assert_collection_exists(db_directory, db_collection)
    except Exception:
        return pd.DataFrame(columns=["id", "source", "page", "document", "embedding"])
    vectorstore = get_chromadb(
        persist_directory=db_directory, collection_name=db_collection
    )
    response = vectorstore.get(include=["metadatas", "documents", "embeddings"])
    return pd.DataFrame(
        {
            "id": response["ids"],
            "source": [metadata.get("source") for metadata in response["metadatas"]],
            "page": [metadata.get("page", -1) for metadata in response["metadatas"]],
            "document": response["documents"],
            "embedding": response["embeddings"],
        }
    )


def get_questions_df(db_directory: Path, db_collection: str) -> pd.DataFrame:
    try:
        _assert_collection_exists(db_directory, db_collection)
    except Exception:
        return pd.DataFrame(
            columns=["id", "question", "answer", "sources", "embedding"]
        )
    vectorstore = get_chromadb(
        persist_directory=db_directory, collection_name=db_collection
    )
    response = vectorstore.get(include=["metadatas", "documents", "embeddings"])
    return pd.DataFrame(
        {
            "id": response["ids"],
            "question": response["documents"],
            "answer": [metadata.get("answer") for metadata in response["metadatas"]],
            "sources": [
                metadata.get("sources").split(",") for metadata in response["metadatas"]
            ],
            "embedding": response["embeddings"],
        }
    )


def get_docs_questions_df(
    docs_db_directory: Path,
    docs_db_collection: str,
    questions_db_directory: Path,
    questions_db_collection: str,
) -> pd.DataFrame:
    docs_df = get_docs_df(docs_db_directory, docs_db_collection)
    docs_df["type"] = "doc"
    questions_df = get_questions_df(questions_db_directory, questions_db_collection)
    questions_df["type"] = "question"

    questions_df["num_sources"] = questions_df["sources"].apply(len)
    questions_df["first_source"] = questions_df["sources"].apply(
        lambda x: next(iter(x), None)
    )

    if len(questions_df):
        docs_df["used_by_questions"] = docs_df["id"].apply(
            lambda doc_id: questions_df[
                questions_df["sources"].apply(lambda sources: doc_id in sources)
            ]["id"].tolist()
        )
    else:
        docs_df["used_by_questions"] = [[] for _ in range(len(docs_df))]
    docs_df["used_by_num_questions"] = docs_df["used_by_questions"].apply(len)
    docs_df["used_by_question_first"] = docs_df["used_by_questions"].apply(
        lambda x: next(iter(x), None)
    )

    df = pd.concat([docs_df, questions_df], ignore_index=True)
    return df
