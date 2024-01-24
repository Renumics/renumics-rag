#!/usr/bin/env python3
import typer

app = typer.Typer()


@app.command()
def answer() -> None:
    ...


if __name__ == "__main__":
    app()
