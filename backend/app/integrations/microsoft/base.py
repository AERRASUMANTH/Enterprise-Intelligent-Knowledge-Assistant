"""Shared interface for Microsoft Graph connectors.

STATUS: scaffolding only. No authentication is performed and no Graph API
request is ever sent. Every data operation raises `ConnectorNotImplementedError`.

Planned design (for a future stage):
- Auth: OAuth 2.0 client-credentials flow (MSAL) against
  https://login.microsoftonline.com/{tenant_id}, scope https://graph.microsoft.com/.default
- Data: Microsoft Graph REST calls under MICROSOFT_GRAPH_BASE_URL.
- Search Agent hook: `search()` will return items in the SAME shape the RAG
  retriever uses ({"text", "source", "page", "distance"}), so results can be
  merged with ChromaDB evidence without changing the agents. It is NOT wired
  into the agent workflow today.
"""

from dataclasses import asdict, dataclass, field

from app.integrations.microsoft.config import MicrosoftGraphSettings, get_microsoft_settings

STATUS_READY = "integration_ready"    # connector interface + config prepared, awaiting credentials
STATUS_COMING_SOON = "coming_soon"    # needs additional Microsoft approval before it can be built


class ConnectorNotImplementedError(NotImplementedError):
    """Raised by every placeholder operation: no live Microsoft data is available."""


@dataclass
class ConnectorInfo:
    key: str
    name: str
    description: str
    capabilities: list[str]
    status: str
    graph_permissions: list[str]           # Microsoft Graph application permissions needed
    graph_endpoints: list[str]             # planned Graph endpoints
    notes: str = ""
    credentials_configured: bool = False   # MICROSOFT_* values present (secrets never exposed)
    authenticated: bool = False            # always False until auth is implemented
    live_data_enabled: bool = False        # always False: no real API calls are made
    extra: dict = field(default_factory=dict)


class GraphConnector:
    """Base class for Outlook / Teams / SharePoint connectors."""

    key = ""
    name = ""
    description = ""
    capabilities: list[str] = []
    status = STATUS_READY
    graph_permissions: list[str] = []
    graph_endpoints: list[str] = []
    notes = ""

    def __init__(self, settings: MicrosoftGraphSettings | None = None):
        self.settings = settings or get_microsoft_settings()

    @property
    def is_authenticated(self) -> bool:
        # Authentication is intentionally not implemented yet.
        return False

    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            key=self.key,
            name=self.name,
            description=self.description,
            capabilities=list(self.capabilities),
            status=self.status,
            graph_permissions=list(self.graph_permissions),
            graph_endpoints=list(self.graph_endpoints),
            notes=self.notes,
            credentials_configured=self.settings.has_credentials,
            authenticated=self.is_authenticated,
            live_data_enabled=False,
        )

    def info_dict(self) -> dict:
        return asdict(self.info())

    def _not_implemented(self, operation: str):
        raise ConnectorNotImplementedError(
            f"{self.name} connector: '{operation}' is not implemented yet. Microsoft Graph "
            "authentication and API calls have not been configured, so no live data is available."
        )

    def search(self, query: str, top: int = 5) -> list[dict]:
        """Future Search Agent hook. Returns retriever-shaped evidence dicts."""
        self._not_implemented("search")
