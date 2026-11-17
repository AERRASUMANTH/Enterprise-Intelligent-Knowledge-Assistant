"""Read-only status of the Microsoft enterprise integrations.

Returns only non-secret metadata and booleans. No authentication, no Graph calls.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.integrations.microsoft import all_connectors
from app.integrations.microsoft.config import get_microsoft_settings

router = APIRouter(tags=["integrations"])


class IntegrationStatus(BaseModel):
    key: str
    name: str
    description: str
    capabilities: list[str]
    status: str
    graph_permissions: list[str]
    graph_endpoints: list[str]
    notes: str
    credentials_configured: bool
    authenticated: bool
    live_data_enabled: bool


class IntegrationsResponse(BaseModel):
    provider: str
    graph_base_url: str
    integrations: list[IntegrationStatus]


@router.get("/integrations", response_model=IntegrationsResponse)
def list_integrations() -> IntegrationsResponse:
    return IntegrationsResponse(
        provider="Microsoft Graph API",
        graph_base_url=get_microsoft_settings().graph_base_url,
        integrations=[IntegrationStatus(**c.info_dict()) for c in all_connectors()],
    )
