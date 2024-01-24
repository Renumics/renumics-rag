#!/usr/bin/env python3
import typer

app = typer.Typer()


@app.command()
def create_db() -> None:
    ...


if __name__ == "__main__":
    app()
