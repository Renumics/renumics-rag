#!/usr/bin/env python3
"""
Explore embeddings database.
"""

import typer

try:
    from renumics import spotlight
    from renumics.spotlight import dtypes as spotlight_dtypes
except ImportError as e:
    raise ImportError(
        "Install Renumics Spotlight to explore vectorstores: "
        "`pip install pandas renumics-spotlight`."
    ) from e

from assistant.exploration import get_docs_questions_df
from assistant.settings import settings

app = typer.Typer()


@app.command()
def explore() -> None:
    """
    Load RAG demo sources and chat history and visualize them.
    """
    df = get_docs_questions_df(
        settings.docs_db_directory,
        settings.docs_db_collection,
        settings.questions_db_directory,
        settings.questions_db_collection,
    )
    print(df)
    spotlight.show(
        df,
        dtype={
            "used_by_questions": spotlight_dtypes.SequenceDType(
                spotlight_dtypes.str_dtype
            )
        },
    )


if __name__ == "__main__":
    app()
