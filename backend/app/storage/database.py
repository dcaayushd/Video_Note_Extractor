from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.models import Base


class DatabaseManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = create_engine(
            settings.database_url,
            connect_args=self._connect_args(settings.database_url),
        )
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    def _connect_args(self, database_url: str) -> dict[str, bool]:
        if database_url.startswith("sqlite:///"):
            return {"check_same_thread": False}
        return {}

    def new_session(self):
        return self.session_factory()
