"""
app/utils/logger.py
Centralised logging setup using Python's standard logging + rich for coloured output.
"""
import logging
import sys
from rich.logging import RichHandler
from rich.console import Console

console = Console()

_configured = False


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger. Call this at the top of every module:
        log = get_logger(__name__)
    """
    global _configured

    if not _configured:
        # Pull level from settings lazily to avoid circular imports at module load
        try:
            from app.config import settings
            level = settings.log_level.upper()
        except Exception:
            level = "INFO"

        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    console=console,
                    rich_tracebacks=True,
                    show_path=True,
                    markup=True,
                )
            ],
        )
        _configured = True

    return logging.getLogger(name)
