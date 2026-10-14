import typer

from evaku_ai.core.config import get_settings
from evaku_ai.core.logging import configure_logging
from evaku_ai.pipeline import orchestrator

app = typer.Typer(help="Evaku data pipeline", no_args_is_help=True)


@app.command(help="Run forecast, radar, write, export, and notify")
def run(mode: orchestrator.Mode = orchestrator.Mode.LIVE) -> None:
    configure_logging(get_settings().log_level)
    orchestrator.run(mode)


@app.command()
def version() -> None:
    from evaku_ai import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
