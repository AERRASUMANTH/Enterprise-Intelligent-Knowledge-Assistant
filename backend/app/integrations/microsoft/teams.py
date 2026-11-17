"""Microsoft Teams connector (placeholder - no live calls).

Marked "Coming Soon": reading channel/chat messages with application permissions
uses Microsoft *protected APIs*, which require a separate approval request to
Microsoft in addition to admin consent.
"""

from app.integrations.microsoft.base import STATUS_COMING_SOON, GraphConnector


class TeamsConnector(GraphConnector):
    key = "teams"
    name = "Microsoft Teams"
    description = "Retrieve and search enterprise team conversations."
    capabilities = ["Message search", "Conversation summarization", "Extract important discussions"]
    status = STATUS_COMING_SOON
    graph_permissions = ["Team.ReadBasic.All", "Channel.ReadBasic.All", "ChannelMessage.Read.All"]
    graph_endpoints = [
        "GET /teams/{team-id}/channels",
        "GET /teams/{team-id}/channels/{channel-id}/messages",
        "GET /teams/{team-id}/channels/{channel-id}/messages/{message-id}/replies",
    ]
    notes = "ChannelMessage.Read.All (application) is a protected API: Microsoft approval is required."

    def list_channels(self, team_id: str) -> list[dict]:
        """Planned: GET /teams/{team_id}/channels"""
        self._not_implemented("list_channels")

    def search_messages(self, query: str, team_id: str, channel_id: str, top: int = 20) -> list[dict]:
        """Planned: GET /teams/{team_id}/channels/{channel_id}/messages, filtered for `query`."""
        self._not_implemented("search_messages")

    def get_thread(self, team_id: str, channel_id: str, message_id: str) -> list[dict]:
        """Planned: GET /teams/{team_id}/channels/{channel_id}/messages/{message_id}/replies"""
        self._not_implemented("get_thread")
