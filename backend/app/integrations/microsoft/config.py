"""Microsoft Graph API configuration, kept separate from the core app settings.

Values come from environment variables or backend/.env (placeholders in .env.example):

    MICROSOFT_CLIENT_ID=
    MICROSOFT_CLIENT_SECRET=
    MICROSOFT_TENANT_ID=
    MICROSOFT_GRAPH_BASE_URL=https://graph.microsoft.com/v1.0

Nothing here is required to run the application: when the values are empty the
connectors simply report that they are not configured.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[3]


class MicrosoftGraphSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MICROSOFT_",
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    client_id: str = ""
    client_secret: SecretStr | None = None   # never logged or returned by the API
    tenant_id: str = ""
    graph_base_url: str = "https://graph.microsoft.com/v1.0"

    @property
    def has_credentials(self) -> bool:
        """True only when all three app-registration values are present."""
        secret = self.client_secret.get_secret_value().strip() if self.client_secret else ""
        return bool(self.client_id.strip() and self.tenant_id.strip() and secret)


@lru_cache
def get_microsoft_settings() -> MicrosoftGraphSettings:
    return MicrosoftGraphSettings()
