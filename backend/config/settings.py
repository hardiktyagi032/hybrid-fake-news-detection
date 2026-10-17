"""Environment-backed settings that never embed real credentials."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_env: str = "development"
    log_level: str = "INFO"
    database_url: str | None = None
    redis_url: str | None = None

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("APP_LOG_LEVEL", "INFO"),
            database_url=os.getenv("DATABASE_URL"),
            redis_url=os.getenv("REDIS_URL"),
        )

    def safe_summary(self) -> dict[str, str | bool]:
        """Expose readiness without revealing connection strings."""

        return {
            "app_env": self.app_env,
            "log_level": self.log_level,
            "database_configured": bool(self.database_url),
            "redis_configured": bool(self.redis_url),
        }
