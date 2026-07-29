"""dosbox-mcp: drive a DOSBox-X guest through MCP, without touching the desktop."""

__all__ = ["main"]


def main() -> None:
    from .server import main as run

    run()
