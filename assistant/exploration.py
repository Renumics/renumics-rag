from pathlib import Path

import pandas as pd

from assistant import get_chromadb


def get_docs_questions_df(
    docs_db_directory: Path,
    docs_db_collection: str,
    questions_db_directory: Path,
    questions_db_collection: str,
) -> pd.DataFrame:
    docs_vectorstore = get_chromadb(
        persist_directory=docs_db_directory, collection_name=docs_db_collection
    )
    response = docs_vectorstore.get(include=["metadatas", "documents", "embeddings"])
    docs_df = pd.DataFrame(
        {
            "id": response["ids"],
            "source": [metadata.get("source") for metadata in response["metadatas"]],
            "page": [metadata.get("page", -1) for metadata in response["metadatas"]],
            "document": response["documents"],
            "embedding": response["embeddings"],
        }
    )

    questions_vectorstore = get_chromadb(
        persist_directory=questions_db_directory,
        collection_name=questions_db_collection,
    )
    response = questions_vectorstore.get(
        include=["metadatas", "documents", "embeddings"]
    )
    questions_df = pd.DataFrame(
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
    questions_df["num_sources"] = questions_df["sources"].apply(len)
    questions_df["first_source"] = questions_df["sources"].apply(
        lambda x: next(iter(x), None)
    )

    docs_df["used_by_questions"] = docs_df["id"].apply(
        lambda doc_id: questions_df[
            questions_df["sources"].apply(lambda sources: doc_id in sources)
        ]["id"].tolist()
    )
    docs_df["used_by_num_questions"] = docs_df["used_by_questions"].apply(len)
    docs_df["used_by_question_first"] = docs_df["used_by_questions"].apply(
        lambda x: next(iter(x), None)
    )

    df = pd.concat([docs_df, questions_df], ignore_index=True)
    return df
