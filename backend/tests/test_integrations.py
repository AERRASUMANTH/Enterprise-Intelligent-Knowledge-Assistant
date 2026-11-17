"""Microsoft integration scaffolding: honest status, no secrets, no live calls."""

import socket
from pathlib import Path

import pytest

from app.integrations.microsoft import (
    STATUS_COMING_SOON,
    STATUS_READY,
    ConnectorNotImplementedError,
    OutlookConnector,
    SharePointConnector,
    TeamsConnector,
)
from app.integrations.microsoft.config import MicrosoftGraphSettings

EMPTY = MicrosoftGraphSettings(_env_file=None, client_id="", tenant_id="", client_secret=None)
FULL = MicrosoftGraphSettings(
    _env_file=None, client_id="test-client", tenant_id="test-tenant", client_secret="super-secret-value"
)


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test if any connector tries to open a network connection."""
    def blocked(*args, **kwargs):
        raise AssertionError("connector attempted a network call")
    monkeypatch.setattr(socket.socket, "connect", blocked)


def test_settings_default_to_unconfigured():
    assert EMPTY.has_credentials is False
    assert EMPTY.graph_base_url == "https://graph.microsoft.com/v1.0"
    assert FULL.has_credentials is True


@pytest.mark.parametrize(
    "connector, calls",
    [
        (OutlookConnector, [("search", ("q",)), ("search_messages", ("q", "user")),
                            ("get_message", ("user", "m1")), ("get_conversation", ("user", "c1"))]),
        (TeamsConnector, [("search", ("q",)), ("list_channels", ("team",)),
                          ("search_messages", ("q", "team", "chan")), ("get_thread", ("team", "chan", "m1"))]),
        (SharePointConnector, [("search", ("q",)), ("search_files", ("q",)),
                               ("list_site_documents", ("site",)), ("download_file", ("drive", "item"))]),
    ],
)
def test_every_operation_is_an_explicit_placeholder(no_network, connector, calls):
    c = connector(settings=FULL)  # even WITH credentials nothing is called
    for name, args in calls:
        with pytest.raises(ConnectorNotImplementedError, match="not implemented yet"):
            getattr(c, name)(*args)


def test_status_is_honest():
    for cls in (OutlookConnector, TeamsConnector, SharePointConnector):
        info = cls(settings=FULL).info()
        assert info.authenticated is False
        assert info.live_data_enabled is False
        assert info.credentials_configured is True
        assert cls(settings=EMPTY).info().credentials_configured is False
    assert OutlookConnector.status == STATUS_READY
    assert SharePointConnector.status == STATUS_READY
    assert TeamsConnector.status == STATUS_COMING_SOON  # protected API approval needed


def test_integrations_endpoint_without_credentials(client):
    body = client.get("/integrations").json()
    assert body["provider"] == "Microsoft Graph API"
    assert body["graph_base_url"] == "https://graph.microsoft.com/v1.0"
    names = [i["name"] for i in body["integrations"]]
    assert names == ["Microsoft Outlook", "Microsoft Teams", "Microsoft SharePoint"]
    for item in body["integrations"]:
        assert item["authenticated"] is False
        assert item["live_data_enabled"] is False
        assert item["status"] in {"integration_ready", "coming_soon"}
        assert item["graph_permissions"]


def test_integrations_endpoint_never_returns_secrets(client, monkeypatch):
    from app.integrations.microsoft import config

    monkeypatch.setattr(config, "get_microsoft_settings", lambda: FULL)
    monkeypatch.setattr("app.integrations.microsoft.base.get_microsoft_settings", lambda: FULL)
    response = client.get("/integrations")
    assert response.status_code == 200
    assert all(i["credentials_configured"] for i in response.json()["integrations"])
    assert "super-secret-value" not in response.text
    assert "test-client" not in response.text and "test-tenant" not in response.text


def test_connectors_are_not_wired_into_agents():
    agents_dir = Path(__file__).resolve().parents[1] / "app" / "agents"
    for py in agents_dir.glob("*.py"):
        assert "integrations" not in py.read_text(encoding="utf-8"), f"{py.name} imports a connector"
