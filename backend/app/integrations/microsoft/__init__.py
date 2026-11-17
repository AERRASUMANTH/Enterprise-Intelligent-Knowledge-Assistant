"""Microsoft Graph connectors: Outlook, Teams, SharePoint.

Readiness scaffolding only - not authenticated, no live API calls, and not used by
the current agent workflow.
"""

from app.integrations.microsoft.base import (
    STATUS_COMING_SOON,
    STATUS_READY,
    ConnectorNotImplementedError,
    GraphConnector,
)
from app.integrations.microsoft.outlook import OutlookConnector
from app.integrations.microsoft.sharepoint import SharePointConnector
from app.integrations.microsoft.teams import TeamsConnector


def all_connectors() -> list[GraphConnector]:
    return [OutlookConnector(), TeamsConnector(), SharePointConnector()]


__all__ = [
    "STATUS_COMING_SOON",
    "STATUS_READY",
    "ConnectorNotImplementedError",
    "GraphConnector",
    "OutlookConnector",
    "SharePointConnector",
    "TeamsConnector",
    "all_connectors",
]
