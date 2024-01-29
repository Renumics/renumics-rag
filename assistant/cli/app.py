#!/usr/bin/env python3
import subprocess
from enum import Enum
from pathlib import Path

import typer
from typing_extensions import Annotated

app = typer.Typer()


class Mode(str, Enum):
    DEV = "development"
    PROD = "production"


@app.command()
def run_app(
    title: Annotated[str, typer.Option(help="Page title")] = "RAG Demo",
    favicon: Annotated[str, typer.Option(help="Page favicon")] = "🤖",
    h1: Annotated[str, typer.Option(help="Title text at the of the page")] = "RAG Demo",
    h2: Annotated[
        str, typer.Option(help="Header text at the top of the page")
    ] = "Chat with your docs",
    mode: Annotated[Mode, typer.Option(help="Mode to start app in.")] = Mode.PROD,
) -> None:
    """Run RAG app."""
    app_filepath = Path(__file__).parent.parent / "app.py"
    args = [
        "streamlit",
        "run",
        "--client.toolbarMode",
        "minimal" if mode == Mode.PROD else "developer",
        str(app_filepath),
        "--",
        "--title",
        title,
        "--favicon",
        favicon,
        "--h1",
        h1,
        "--h2",
        h2,
    ]
    subprocess.run(args)


if __name__ == "__main__":
    app()
