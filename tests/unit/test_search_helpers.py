"""Unit tests for search helpers.

Covers reciprocal rank fusion, deduplication, query cleaning, search routing, and embedding lookup by ID.
"""

import importlib
import sys
import types


def _ensure_service_import_stubs():
    """Install lightweight stubs into system modules for unit tests without real external dependencies."""
    if "requests" not in sys.modules:
        fake_requests = types.ModuleType("requests")
        fake_requests.get = lambda *args, **kwargs: None
        fake_requests.post = lambda *args, **kwargs: None
        sys.modules["requests"] = fake_requests

    if "stopwordsiso" not in sys.modules:
        fake_stopwordsiso = types.ModuleType("stopwordsiso")
        fake_stopwordsiso.stopwords = lambda _lang: {"the", "a", "an"}
        sys.modules["stopwordsiso"] = fake_stopwordsiso

    if "astrapy" not in sys.modules:
        fake_astrapy = types.ModuleType("astrapy")
        fake_astrapy.DataAPIClient = object
        sys.modules["astrapy"] = fake_astrapy

    if "astrapy.api_options" not in sys.modules:
        fake_api_options = types.ModuleType("astrapy.api_options")
        fake_api_options.APIOptions = object
        fake_api_options.TimeoutOptions = object
        sys.modules["astrapy.api_options"] = fake_api_options

    if "wikidatasearch.services.jina" not in sys.modules:
        fake_jina = types.ModuleType("wikidatasearch.services.jina")

        class _DummyJina:
            """Minimal Jina client stub."""

            def __init__(self, *_args, **_kwargs):
                """Accept arbitrary constructor args in tests."""
                pass

        fake_jina.JinaAIAPI = _DummyJina
        sys.modules["wikidatasearch.services.jina"] = fake_jina


def _service_classes():
    """Import and return search service classes with the dependency stubs."""
    _ensure_service_import_stubs()

    hybrid_module = importlib.import_module("wikidatasearch.services.search.HybridSearch")
    keyword_module = importlib.import_module("wikidatasearch.services.search.KeywordSearch")
    vector_module = importlib.import_module("wikidatasearch.services.search.VectorSearch")

    return hybrid_module.HybridSearch, keyword_module.KeywordSearch, vector_module.VectorSearch


def test_reciprocal_rank_fusion_merges_sources_and_accumulates_rrf(test_ctx):
    """Validate reciprocal rank fusion sources and accumulates rrf score."""
    HybridSearch, _, _ = _service_classes()
    vector_results = [
        {"QID": "Q42", "similarity_score": 0.95},
        {"QID": "Q5", "similarity_score": 0.90},
    ]
    keyword_results = [
        {"QID": "Q5", "similarity_score": 0.80},
        {"QID": "Q42", "similarity_score": 0.70},
    ]

    fused = HybridSearch.reciprocal_rank_fusion(
        [
            ("Vector Search", vector_results),
            ("Keyword Search", keyword_results),
        ]
    )

    by_id = {row["QID"]: row for row in fused}
    assert set(by_id.keys()) == {"Q42", "Q5"}
    assert by_id["Q42"]["source"] == "Vector Search, Keyword Search"
    assert by_id["Q5"]["source"] == "Vector Search, Keyword Search"
    assert by_id["Q42"]["similarity_score"] == 0.95
    assert by_id["Q5"]["similarity_score"] == 0.90
    assert by_id["Q42"]["rrf_score"] > 0
    assert by_id["Q5"]["rrf_score"] > 0


def test_vector_remove_duplicates_prefers_best_similarity_and_keeps_unique_results(test_ctx):
    """Validate removing duplicates that keeps unique results with the highest similarity scores."""
    _, _, VectorSearch = _service_classes()
    raw_results = [
        {"metadata": {"QID": "Q42"}, "$similarity": 0.60, "$vector": [0.1], "content": "A"},
        {"metadata": {"QID": "Q42"}, "$similarity": 0.95, "$vector": [0.9], "content": "B"},
        {"metadata": {"PID": "P31"}, "$similarity": 0.70, "$vector": [0.2], "content": "C"},
    ]

    deduped = VectorSearch.remove_duplicates(
        raw_results,
        return_vectors=True,
        return_text=True,
    )

    assert len(deduped) == 2
    assert deduped[0]["QID"] == "Q42"
    assert deduped[0]["similarity_score"] == 0.95
    assert deduped[0]["vector"] == [0.9]
    assert deduped[0]["text"] == "B"
    assert deduped[1]["PID"] == "P31"
    assert deduped[1]["similarity_score"] == 0.70


def test_reciprocal_rank_fusion_drops_non_positive_similarity(test_ctx):
    """Validate reciprocal rank fusion that drops negative similarity scores."""
    HybridSearch, _, _ = _service_classes()
    fused = HybridSearch.reciprocal_rank_fusion(
        [
            (
                "Vector Search",
                [
                    {"QID": "Q3", "similarity_score": 0.25},
                    {"QID": "Q1", "similarity_score": 0.0},
                    {"QID": "Q2", "similarity_score": -0.1},
                ],
            )
        ]
    )

    assert [row["QID"] for row in fused] == ["Q3"]


def test_keyword_clean_query_removes_stopwords_and_caps_length(test_ctx):
    """Validate KeywordSearch's clean query that removes stopwords and caps length."""
    _, KeywordSearch, _ = _service_classes()
    keyword = KeywordSearch()

    cleaned = keyword._clean_query("the quick brown fox", "all")
    assert "the" not in cleaned.lower()
    assert "quick" in cleaned.lower()
    assert len(cleaned) <= 300

    very_long = "word " * 500
    cleaned_long = keyword._clean_query(very_long, "en")
    assert len(cleaned_long) <= 300


def test_keyword_search_filters_external_id_properties_after_cirrus_search(test_ctx, monkeypatch):
    """Validate keyword property search preserves Cirrus search and filters datatypes after."""
    _, KeywordSearch, _ = _service_classes()
    keyword_module = importlib.import_module("wikidatasearch.services.search.KeywordSearch")
    calls = []

    class _Response:
        """Minimal response stub."""

        def __init__(self, payload):
            """Store the JSON payload."""
            self.payload = payload

        def raise_for_status(self):
            """Match the requests response API used by the search code."""
            return None

        def json(self):
            """Return the configured JSON payload."""
            return self.payload

    def _fake_get(url, params=None, headers=None):
        """Return Cirrus hits first, then property datatype metadata."""
        calls.append({"url": url, "params": params, "headers": headers})
        if url.endswith("/w/index.php"):
            return _Response(
                {
                    "__main__": {
                        "result": {
                            "hits": {
                                "hits": [
                                    {"_source": {"title": "P214"}},
                                    {"_source": {"title": "P31"}},
                                ]
                            }
                        }
                    }
                }
            )

        return _Response(
            {
                "entities": {
                    "P214": {"type": "property", "datatype": "external-id", "id": "P214"},
                    "P31": {"type": "property", "datatype": "wikibase-item", "id": "P31"},
                }
            }
        )

    monkeypatch.setattr(keyword_module.requests, "get", _fake_get)

    keyword = KeywordSearch()
    results = keyword.search(
        "instance",
        filter={
            "metadata.IsProperty": True,
            "metadata.DataType": {"$ne": "external-id"},
        },
        K=2,
    )

    assert results == ["P31"]
    assert calls[0]["url"] == "https://www.wikidata.org/w/index.php"
    assert calls[0]["params"]["srlimit"] == 2
    assert calls[1]["url"] == "https://www.wikidata.org/w/api.php"
    assert calls[1]["params"]["action"] == "wbgetentities"
    assert calls[1]["params"]["ids"] == "P214|P31"


def test_keyword_search_returns_direct_pid_regardless_of_filter(test_ctx):
    """Validate direct PID searches bypass result filters."""
    _, KeywordSearch, _ = _service_classes()

    keyword = KeywordSearch()
    results = keyword.search(
        "P214",
        filter={
            "metadata.IsProperty": True,
            "metadata.DataType": {"$ne": "external-id"},
        },
        K=1,
    )

    assert results == ["P214"]


def test_keyword_property_datatype_lookup_batches_ids(test_ctx, monkeypatch):
    """Validate datatype lookups are split into Wikidata API-sized batches."""
    _, KeywordSearch, _ = _service_classes()
    keyword_module = importlib.import_module("wikidatasearch.services.search.KeywordSearch")
    calls = []

    class _Response:
        """Minimal response stub."""

        def __init__(self, ids):
            """Store the requested property IDs."""
            self.ids = ids

        def raise_for_status(self):
            """Match the requests response API used by the search code."""
            return None

        def json(self):
            """Return datatype metadata for requested properties."""
            return {"entities": {pid: {"datatype": "wikibase-item"} for pid in self.ids}}

    def _fake_get(url, params=None, headers=None):
        """Capture batched datatype requests."""
        ids = params["ids"].split("|")
        calls.append(ids)
        return _Response(ids)

    monkeypatch.setattr(keyword_module.requests, "get", _fake_get)

    keyword = KeywordSearch()
    datatypes = keyword._get_property_datatypes([f"P{i}" for i in range(1, 52)])

    assert len(calls) == 2
    assert len(calls[0]) == 50
    assert calls[1] == ["P51"]
    assert datatypes["P1"] == "wikibase-item"
    assert datatypes["P51"] == "wikibase-item"


def test_vector_find_uses_configured_collection(test_ctx):
    """Validate VectorSearch always queries its configured collection."""
    _, _, VectorSearch = _service_classes()

    class _FakeCollection:
        """Minimal collection stub that records find calls."""

        def __init__(self):
            """Initialize captured calls."""
            self.calls = []

        def find(self, *args, **kwargs):
            """Capture call arguments and return one deterministic row."""
            self.calls.append({"args": args, "kwargs": kwargs})
            return [{"metadata": {"PID": "P31"}, "$similarity": 0.9}]

    vector = VectorSearch.__new__(VectorSearch)
    vector.collection = _FakeCollection()
    vector.max_K = 50

    rows = vector.find(
        {"metadata.PID": {"$in": ["P31"]}},
        projection={"metadata": 1},
        limit=None,
    )

    assert rows and rows[0]["metadata"]["PID"] == "P31"
    assert len(vector.collection.calls) == 1
    assert vector.collection.calls[0]["args"][0] == {"metadata.PID": {"$in": ["P31"]}}


def test_get_embedding_by_id_uses_configured_id_field(test_ctx):
    """Validate that ID lookups use the field configured for the collection."""
    _, _, VectorSearch = _service_classes()

    captured = {}

    def _fake_find(filter, projection=None, limit=50, sort=None, include_similarity=True):
        """Capture incoming filter and return one vector row."""
        captured.update(filter)
        return [{"metadata": {"PID": "P31"}, "$vector": [0.1, 0.2]}]

    vector = VectorSearch.__new__(VectorSearch)
    vector.id_field = "PID"
    vector.find = _fake_find

    item, embedding = vector.get_embedding_by_id("P31")

    assert item["metadata"]["PID"] == "P31"
    assert embedding == [0.1, 0.2]
    assert captured == {"metadata.PID": "P31"}
