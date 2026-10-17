"""Repository setup status.

This is deliberately not a FastAPI application. Application endpoints remain outside
the authorized Phase 1-4 scope.
"""

from __future__ import annotations

from backend import __version__


def setup_status() -> dict[str, str]:
    """Return machine-readable repository status for initial verification."""

    return {
        "project": "hybrid-fake-news-detection",
        "version": __version__,
        "phase": "waiting-for-user-dataset",
        "application_implemented": "false",
    }


if __name__ == "__main__":
    print(setup_status())
