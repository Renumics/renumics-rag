#!/usr/bin/env python3
"""
Explore embeddings database.
"""

try:
    import pandas as pd
    from renumics import spotlight
    from renumics.spotlight import dtypes as spotlight_dtypes
except ImportError as e:
    raise ImportError(
        "In order to explore vectorstores, install extra packages: "
        "`pip install pandas renumics-spotlight`."
    ) from e
import typer
from typing_extensions import Annotated

from assistant import get_chromadb, get_embeddings_model, parse_model_name
from assistant.const import EMBEDDINGS_MODEL_NAME_HELP
from assistant.settings import settings

app = typer.Typer()


@app.command()
def explore(
    embeddings_model_name: Annotated[
        str, typer.Option("--embeddings", help=EMBEDDINGS_MODEL_NAME_HELP)
    ] = settings.full_embeddings_model_name,
) -> None:
    """
    Load RAG demo sources and chat history and visalize them.
    """

    embeddings_model = get_embeddings_model(*parse_model_name(embeddings_model_name))

    docs_vectorstore = get_chromadb(
        settings.docs_db_directory, embeddings_model, settings.docs_db_collection
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
        settings.questions_db_directory,
        embeddings_model,
        settings.questions_db_collection,
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
    print(df)
    spotlight.show(
        df,
        dtype={
            "used_by_questions": spotlight_dtypes.SequenceDType(
                spotlight_dtypes.str_dtype
            )
        },
        embed=False,
        analyze=False,
    )


if __name__ == "__main__":
    app()
