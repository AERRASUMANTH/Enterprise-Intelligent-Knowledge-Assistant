from app.services.indexer import index_document
from app.services.retriever import retrieve
from app.services.vector_store import VectorStore
from tests.conftest import make_pdf

INCIDENT = (
    "Incident report. The primary database server failed due to a full disk. "
    "The issue was resolved by clearing old log files and adding disk monitoring."
)
HOLIDAY = "Company holiday schedule. The office is closed on national holidays."


def test_empty_store_returns_no_results(embedder, store):
    assert retrieve("anything?", 5, embedder, store) == []


def test_insert_and_retrieve(embedder, store):
    index_document("incident.txt", INCIDENT.encode(), embedder, store)
    index_document("holidays.txt", HOLIDAY.encode(), embedder, store)

    results = retrieve("How was the server failure resolved?", 5, embedder, store)

    assert len(results) == 2  # only 2 chunks exist, even though top_k=5
    assert results[0]["source"] == "incident.txt"
    assert "disk" in results[0]["text"]
    assert results[0]["distance"] < results[1]["distance"]


def test_top_k_limits_results(embedder, store):
    for i in range(4):
        index_document(f"doc{i}.txt", f"Document number {i} text.".encode(), embedder, store)
    assert len(retrieve("document", 2, embedder, store)) == 2


def test_pdf_page_numbers_are_returned(embedder, store):
    pdf = make_pdf(["Cafeteria opening hours", "The VPN outage was fixed by renewing certificates"])
    index_document("ops.pdf", pdf, embedder, store)

    top = retrieve("How was the VPN outage fixed?", 1, embedder, store)[0]
    assert top["source"] == "ops.pdf"
    assert top["page"] == 2


def test_same_document_uploaded_twice_is_not_duplicated(embedder, store):
    first = index_document("incident.txt", INCIDENT.encode(), embedder, store)
    count_after_first = store.count()
    second = index_document("incident.txt", INCIDENT.encode(), embedder, store)

    assert first.status == "indexed"
    assert second.status == "unchanged"
    assert store.count() == count_after_first


def test_changed_document_replaces_old_chunks(embedder, store):
    index_document("policy.txt", ("old policy text " * 200).encode(), embedder, store)
    result = index_document("policy.txt", b"new short policy", embedder, store)

    assert result.status == "replaced"
    assert store.count() == 1
    assert retrieve("policy", 5, embedder, store)[0]["text"] == "new short policy"


def test_retrieval_is_semantic_not_keyword(embedder, store):
    # The question shares no content words with the matching document.
    index_document("vpn.txt", b"Remote access was restored after the expired certificate was renewed.", embedder, store)
    index_document("food.txt", b"The cafeteria serves pasta and salad on Mondays.", embedder, store)
    index_document("hr.txt", b"Annual leave requests must be submitted two weeks in advance.", embedder, store)

    top = retrieve("How did they repair the VPN connectivity problem?", 3, embedder, store)[0]
    assert top["source"] == "vpn.txt"


def test_irrelevant_chunks_filtered_by_distance(embedder, store):
    index_document("incident.txt", INCIDENT.encode(), embedder, store)
    index_document("holidays.txt", HOLIDAY.encode(), embedder, store)

    everything = retrieve("database disk failure", 5, embedder, store, max_distance=2.0)
    strict = retrieve("database disk failure", 5, embedder, store, max_distance=0.5)

    assert len(everything) == 2
    assert [r["source"] for r in strict] == ["incident.txt"]
    assert all(r["distance"] <= 0.5 for r in strict)


def test_data_persists_across_instances(embedder, tmp_path):
    path = tmp_path / "persist"
    index_document("incident.txt", INCIDENT.encode(), embedder, VectorStore(path, "persist_test"))

    reopened = VectorStore(path, "persist_test")
    assert reopened.count() == 1
    assert retrieve("disk", 1, embedder, reopened)[0]["source"] == "incident.txt"
