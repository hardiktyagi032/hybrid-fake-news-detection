from __future__ import annotations

from pathlib import Path

from backend.config.settings import Settings
from backend.main import setup_status


ROOT = Path(__file__).resolve().parents[1]


def test_setup_status_is_honest_about_scope() -> None:
    status = setup_status()
    assert status["phase"] == "waiting-for-user-dataset"
    assert status["application_implemented"] == "false"


def test_required_repository_boundaries_exist() -> None:
    expected = [
        "backend/api",
        "backend/preprocessing",
        "backend/ml",
        "backend/rag",
        "backend/nli",
        "backend/credibility",
        "backend/auxiliary",
        "backend/fusion",
        "backend/llm",
        "backend/database",
        "backend/cache",
        "data/raw",
        "data/interim",
        "data/processed",
        "docs/architecture",
        "docs/research",
        "docs/data",
        "frontend",
        "scripts",
    ]
    missing = [item for item in expected if not (ROOT / item).exists()]
    assert missing == []


def test_settings_summary_does_not_expose_connection_values(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret")
    monkeypatch.setenv("REDIS_URL", "redis://secret")
    summary = Settings.from_environment().safe_summary()
    assert summary["database_configured"] is True
    assert summary["redis_configured"] is True
    assert "postgresql://secret" not in str(summary)
    assert "redis://secret" not in str(summary)
