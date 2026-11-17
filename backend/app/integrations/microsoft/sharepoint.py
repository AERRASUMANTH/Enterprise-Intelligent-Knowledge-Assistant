"""Microsoft SharePoint connector (placeholder - no live calls).

Future indexing path: download a file's content, then hand it to the EXISTING
ingestion pipeline (app.services.indexer.index_document) - no second RAG pipeline.
"""

from app.integrations.microsoft.base import STATUS_READY, GraphConnector


class SharePointConnector(GraphConnector):
    key = "sharepoint"
    name = "Microsoft SharePoint"
    description = "Access enterprise documents and knowledge repositories."
    capabilities = ["Document retrieval", "Indexing into the knowledge base", "Search SharePoint content"]
    status = STATUS_READY
    graph_permissions = ["Sites.Read.All", "Files.Read.All"]
    graph_endpoints = [
        "POST /search/query  (entityTypes: driveItem, listItem)",
        "GET /sites/{site-id}/drive/root/children",
        "GET /drives/{drive-id}/items/{item-id}/content",
    ]
    notes = "Sites.Selected can replace Sites.Read.All to limit access to approved sites."

    def search_files(self, query: str, top: int = 10) -> list[dict]:
        """Planned: POST /search/query with entityTypes ["driveItem"]"""
        self._not_implemented("search_files")

    def list_site_documents(self, site_id: str) -> list[dict]:
        """Planned: GET /sites/{site_id}/drive/root/children"""
        self._not_implemented("list_site_documents")

    def download_file(self, drive_id: str, item_id: str) -> bytes:
        """Planned: GET /drives/{drive_id}/items/{item_id}/content"""
        self._not_implemented("download_file")
