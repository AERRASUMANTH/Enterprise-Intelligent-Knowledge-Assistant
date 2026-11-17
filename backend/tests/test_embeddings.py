import math

from app.services.embeddings import get_embedding_service


def test_embeddings_have_consistent_dimension(embedder):
    vectors = embedder.embed_texts(["first text", "second text"])
    query = embedder.embed_query("a question")

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) == len(query) > 0


def test_embeddings_are_normalized(embedder):
    vector = embedder.embed_query("normalize me")
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-3)


def test_similar_texts_are_closer(embedder):
    q, related, unrelated = embedder.embed_texts(
        ["How do I reset my password?", "Steps to reset a forgotten password", "Lunch menu for Friday"]
    )
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(q, related) > dot(q, unrelated)


def test_empty_batch(embedder):
    assert embedder.embed_texts([]) == []


def test_service_is_loaded_once():
    get_embedding_service.cache_clear()
    try:
        assert get_embedding_service() is get_embedding_service()
    finally:
        get_embedding_service.cache_clear()
