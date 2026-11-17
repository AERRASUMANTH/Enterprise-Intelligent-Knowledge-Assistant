"""Microsoft Outlook connector (placeholder - no live calls)."""

from app.integrations.microsoft.base import STATUS_READY, GraphConnector


class OutlookConnector(GraphConnector):
    key = "outlook"
    name = "Microsoft Outlook"
    description = "Retrieve and search enterprise emails."
    capabilities = ["Email search", "Email summarization", "Find relevant conversations"]
    status = STATUS_READY
    graph_permissions = ["Mail.Read", "User.Read.All"]
    graph_endpoints = [
        "GET /users/{user-id}/messages?$search=\"{query}\"",
        "GET /users/{user-id}/messages/{message-id}",
        "GET /users/{user-id}/mailFolders",
    ]
    notes = "Mail.Read (application) needs tenant admin consent; scope it with an application access policy."

    def search_messages(self, query: str, user_id: str, top: int = 10) -> list[dict]:
        """Planned: GET /users/{user_id}/messages?$search="{query}"&$top={top}"""
        self._not_implemented("search_messages")

    def get_message(self, user_id: str, message_id: str) -> dict:
        """Planned: GET /users/{user_id}/messages/{message_id}"""
        self._not_implemented("get_message")

    def get_conversation(self, user_id: str, conversation_id: str) -> list[dict]:
        """Planned: GET /users/{user_id}/messages?$filter=conversationId eq '{conversation_id}'"""
        self._not_implemented("get_conversation")
