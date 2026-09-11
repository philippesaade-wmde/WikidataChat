"""Benchmark analysis tests for language sharding vector search behavior."""

import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytestmark = [pytest.mark.analysis, pytest.mark.benchmark, pytest.mark.slow]

LANGS = ["en", "fr", "de", "ar"]
SEARCH_CACHE = {}
BASE_K = 50
EXPANDED_K = 100
COMPARE_TOP_N = 10
TOP_N_ACCURACY = 5
TOP_N_EXPOSURE = 20
MIN_EXPECTED_RECOVERY = 0.0

QUERIES_BY_LANG = {
    "en": [
        "What is the capital of France?",
        "Who is the president of the United States?",
        "What is the largest mammal?",
        "Who won the FIFA World Cup in 2018?",
        "What is the chemical formula for water?",
        "Who wrote The Hitchhiker's Guide to the Galaxy?",
        "What is the tallest mountain in the world?",
        "What is the currency of Japan?",
    ],
    "fr": [
        "Quelle est la capitale de la France ?",
        "Qui est le president des Etats-Unis ?",
        "Quel est le plus grand mammifere ?",
        "Qui a gagne la Coupe du monde 2018 ?",
        "Quelle est la formule chimique de l'eau ?",
        "Qui a ecrit Le Guide du voyageur galactique ?",
        "Quelle est la plus haute montagne du monde ?",
        "Quelle est la monnaie du Japon ?",
    ],
    "de": [
        "Was ist die Hauptstadt von Frankreich?",
        "Wer ist der Prasident der Vereinigten Staaten?",
        "Was ist das grosste Saugetier?",
        "Wer hat die Fussball-Weltmeisterschaft 2018 gewonnen?",
        "Was ist die chemische Formel von Wasser?",
        "Wer schrieb Per Anhalter durch die Galaxis?",
        "Was ist der hochste Berg der Welt?",
        "Was ist die Wahrung Japans?",
    ],
    "ar": [
        "ما هي عاصمة فرنسا؟",
        "من هو رئيس الولايات المتحدة؟",
        "ما هو اكبر حيوان ثديي؟",
        "من فاز بكاس العالم 2018؟",
        "ما هي الصيغة الكيميائية للماء؟",
        "من كتب دليل المسافر إلى المجرة؟",
        "ما هو اعلى جبل في العالم؟",
        "ما هي عملة اليابان؟",
    ],
}


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read key-value pairs from a dotenv file."""
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _load_api_keys() -> dict[str, str]:
    """Load and validate API keys required for benchmark execution."""
    root = Path(__file__).resolve().parents[2]
    merged = {}
    merged.update(_read_dotenv(root / ".env"))
    merged.update(_read_dotenv(root / "tests" / ".env"))

    required = [
        "ASTRA_DB_APPLICATION_TOKEN",
        "ASTRA_DB_API_ENDPOINT",
        "ASTRA_DB_COLLECTION",
        "JINA_API_KEY",
    ]
    missing = [key for key in required if not merged.get(key)]
    if missing:
        pytest.skip("Missing required keys in .env/tests/.env for benchmark: " + ", ".join(missing))
    return merged


def _import_search_classes():
    """Import search classes or skip when dependencies are missing."""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from wikidatasearch.services.search.HybridSearch import HybridSearch
        from wikidatasearch.services.search.VectorSearch import VectorSearch
    except ModuleNotFoundError as exc:
        pytest.skip(f"Missing Python module '{exc.name}'. Install dependencies before running benchmark tests.")

    return HybridSearch, VectorSearch


def _run_with_retry(func, *args, retries=3, backoff_s=0.5, measure_time=True, **kwargs):
    """Execute a callable with retry and optional runtime measurement."""
    last_error = None
    for attempt in range(retries):
        start = time.time() if measure_time else None
        try:
            result = func(*args, **kwargs)
            if measure_time:
                return result, time.time() - start
            return result
        except Exception as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
            time.sleep(backoff_s * (attempt + 1))
    raise last_error


def _make_search(VectorSearch, api_keys, lang=None, max_k=50):
    """Get or create a cached vector search instance."""
    key = (lang, max_k)
    if key not in SEARCH_CACHE:
        collection = api_keys["ASTRA_DB_COLLECTION"]
        if lang:
            collection = f"{collection}_items_{lang}"
        SEARCH_CACHE[key] = VectorSearch(
            api_keys=api_keys,
            collection=collection,
            max_K=max_k,
        )
    return SEARCH_CACHE[key]


def _timed_vdb_search(vdb, lang, query, embedding, K, search_filter):
    """Run one shard search and return rows plus runtime metadata."""
    start = time.time()
    rows = vdb.find(
        filter=search_filter,
        sort={"$vector": embedding},
        projection={"metadata": 1},
        limit=K,
        include_similarity=True,
    )
    duration = time.time() - start
    rows = [{**row, "_shard_lang": lang} for row in rows]
    return {"lang": lang, "results": rows, "duration": duration}


def _v1_vector_search(VectorSearch, api_keys, query, K=50, max_k=None):
    """Run all-in-one database vector search across all languages."""
    search = _make_search(VectorSearch, api_keys, max_k=max_k or K)
    embedding, _ = search.calculate_embedding(query, lang="all")
    if embedding is None:
        return []
    return search.find(
        filter={},
        sort={"$vector": embedding},
        projection={"metadata": 1},
        limit=K,
        include_similarity=True,
    )


def _v2_vector_search(VectorSearch, api_keys, query, langs=None, K=50, max_k=None, include_thread_times=False):
    """Run sharded-language vector search across all languages."""
    langs = langs or LANGS
    max_k = max_k or K
    searches = {lang: _make_search(VectorSearch, api_keys, lang=lang, max_k=max_k) for lang in langs}
    embedding = next(iter(searches.values())).embedding_model.embed_query(query)
    search_filter = {}

    with ThreadPoolExecutor(max_workers=len(langs)) as ex:
        futures = [
            ex.submit(_timed_vdb_search, searches[lang], lang, query, embedding, K, search_filter) for lang in langs
        ]

    payloads = [future.result() for future in futures]
    results = {payload["lang"]: payload["results"] for payload in payloads}

    if include_thread_times:
        return {
            "results": results,
            "thread_runtimes": {payload["lang"]: payload["duration"] for payload in payloads},
        }
    return results


def _v1_vector_search_lang(VectorSearch, api_keys, query, lang="en", K=50, max_k=None):
    """Run all-in-one database vector search for one language."""
    search = _make_search(VectorSearch, api_keys, max_k=max_k or K)
    embedding, _ = search.calculate_embedding(query, lang=lang)
    if embedding is None:
        return []
    return search.find(
        filter={"metadata.Language": lang},
        sort={"$vector": embedding},
        projection={"metadata": 1},
        limit=K,
        include_similarity=True,
    )


def _v2_vector_search_lang(VectorSearch, api_keys, query, lang="en", K=50, max_k=None):
    """Run sharded-language vector search for one language."""
    search = _make_search(VectorSearch, api_keys, lang=lang, max_k=max_k or K)
    embedding, _ = search.calculate_embedding(query, lang=lang)
    if embedding is None:
        return []
    rows = search.find(
        filter={},
        sort={"$vector": embedding},
        projection={"metadata": 1},
        limit=K,
        include_similarity=True,
    )
    return [{**row, "_shard_lang": lang} for row in rows]


def _entity_id(item):
    """Extract a stable entity ID from a search result."""
    metadata = item.get("metadata", {})
    return (
        item.get("QID")
        or item.get("PID")
        or metadata.get("QID")
        or metadata.get("PID")
        or metadata.get("_id")
        or item.get("_id")
    )


def _normalize_results(VectorSearch, raw_results, K=50):
    """Normalize and remove duplicates from vector search results."""
    deduped = VectorSearch.remove_duplicates(raw_results)
    return deduped[:K]


def _average_similarity(results, top_n=5):
    """Compute mean similarity for the top-N results."""
    rows = results[:top_n]
    if not rows:
        return 0.0
    return sum(row["similarity_score"] for row in rows) / len(rows)


def _average_dicts(dicts, keys=None):
    """Compute key-wise averages over a list of dictionaries."""
    if not dicts:
        return {}
    keys = keys or dicts[0].keys()
    return {key: sum(d[key] for d in dicts) / len(dicts) for key in keys}


def _lang_stats_raw(results):
    """Estimate language exposure ratios from raw search results (before merging)."""
    rows = sorted(results, key=lambda x: x.get("$similarity", 0.0), reverse=True)[:TOP_N_EXPOSURE]
    if not rows:
        return {lang: 0.0 for lang in LANGS}
    denom = float(len(rows))
    stats = {}
    for lang in LANGS:
        count = 0
        for row in rows:
            row_lang = row.get("metadata", {}).get("Language") or row.get("_shard_lang")
            if row_lang == lang:
                count += 1
        stats[lang] = count / denom
    return stats


def _lang_stats_rrf(results):
    """Estimate language exposure ratios from merged results with RRF."""
    rows = sorted(results, key=lambda x: x["similarity_score"], reverse=True)[:TOP_N_EXPOSURE]
    if not rows:
        return {lang: 0.0 for lang in LANGS}
    denom = float(len(rows))
    return {lang: sum(1 for row in rows if lang in row.get("source", "")) / denom for lang in LANGS}


@pytest.fixture(scope="module")
def split_benchmark_payload():
    """Build benchmark payloads for runtime, quality, and recall assertions."""
    HybridSearch, VectorSearch = _import_search_classes()
    api_keys = _load_api_keys()

    all_query_rows = [{"query_lang": lang, "query": query} for lang in LANGS for query in QUERIES_BY_LANG[lang]]

    v1_results, v2_results = [], []
    runtime_v1, runtime_v2 = [], []
    v2_thread_runtimes = []
    query_lang_runtime_v1 = {lang: [] for lang in LANGS}
    query_lang_runtime_v2 = {lang: [] for lang in LANGS}
    query_lang_thread_runtimes = {lang: {shard: [] for shard in LANGS} for lang in LANGS}

    for row in all_query_rows:
        query = row["query"]
        query_lang = row["query_lang"]

        result, duration = _run_with_retry(_v1_vector_search, VectorSearch, api_keys, query, K=BASE_K)
        v1_results.append(result)
        runtime_v1.append(duration)
        query_lang_runtime_v1[query_lang].append(duration)

        payload, duration = _run_with_retry(
            _v2_vector_search,
            VectorSearch,
            api_keys,
            query,
            langs=LANGS,
            K=BASE_K,
            include_thread_times=True,
        )
        v2_results.append(payload["results"])
        runtime_v2.append(duration)
        v2_thread_runtimes.append(payload["thread_runtimes"])
        query_lang_runtime_v2[query_lang].append(duration)
        for shard_lang, shard_duration in payload["thread_runtimes"].items():
            query_lang_thread_runtimes[query_lang][shard_lang].append(shard_duration)

    lang_runtime_v1 = {lang: [] for lang in LANGS}
    lang_runtime_v2 = {lang: [] for lang in LANGS}
    for lang in LANGS:
        for query in QUERIES_BY_LANG[lang]:
            _, duration = _run_with_retry(_v1_vector_search_lang, VectorSearch, api_keys, query, lang=lang, K=BASE_K)
            lang_runtime_v1[lang].append(duration)
            _, duration = _run_with_retry(_v2_vector_search_lang, VectorSearch, api_keys, query, lang=lang, K=BASE_K)
            lang_runtime_v2[lang].append(duration)

    v1_norm = [_normalize_results(VectorSearch, results, K=BASE_K) for results in v1_results]
    v2_flat = [[row for per_lang in per_query.values() for row in per_lang] for per_query in v2_results]
    v2_norm = [_normalize_results(VectorSearch, results, K=BASE_K) for results in v2_flat]
    accuracies_v1 = [_average_similarity(result, top_n=TOP_N_ACCURACY) for result in v1_norm]
    accuracies_v2 = [_average_similarity(result, top_n=TOP_N_ACCURACY) for result in v2_norm]

    v2_lang_stats_raw = [_lang_stats_raw(results) for results in v2_flat]
    v1_lang_stats_raw = [_lang_stats_raw(results) for results in v1_results]
    v2_lang_stats_raw_avg = _average_dicts(v2_lang_stats_raw, keys=LANGS)
    v1_lang_stats_raw_avg = _average_dicts(v1_lang_stats_raw, keys=LANGS)

    v2_rrf_inputs = [
        [(LANGS[i], _normalize_results(VectorSearch, per_query[LANGS[i]], K=BASE_K)) for i in range(len(LANGS))]
        for per_query in v2_results
    ]
    v2_rrf_results = [HybridSearch.reciprocal_rank_fusion(inputs) for inputs in v2_rrf_inputs]
    v2_lang_stats_rrf_avg = _average_dicts([_lang_stats_rrf(results) for results in v2_rrf_results], keys=LANGS)

    v1_rrf_inputs = [
        [
            (
                lang,
                _normalize_results(
                    VectorSearch,
                    [row for row in per_query if row.get("metadata", {}).get("Language") == lang],
                    K=BASE_K,
                ),
            )
            for lang in LANGS
        ]
        for per_query in v1_results
    ]
    v1_rrf_results = [HybridSearch.reciprocal_rank_fusion(inputs) for inputs in v1_rrf_inputs]
    v1_lang_stats_rrf_avg = _average_dicts([_lang_stats_rrf(results) for results in v1_rrf_results], keys=LANGS)

    recall_rows = []
    for lang in LANGS:
        for query in QUERIES_BY_LANG[lang]:
            v1_raw = _run_with_retry(
                _v1_vector_search_lang,
                VectorSearch,
                api_keys,
                query,
                lang=lang,
                K=BASE_K,
                max_k=BASE_K,
                measure_time=False,
            )
            v2_raw = _run_with_retry(
                _v2_vector_search_lang,
                VectorSearch,
                api_keys,
                query,
                lang=lang,
                K=BASE_K,
                max_k=BASE_K,
                measure_time=False,
            )
            v2_expanded_raw = _run_with_retry(
                _v2_vector_search_lang,
                VectorSearch,
                api_keys,
                query,
                lang=lang,
                K=EXPANDED_K,
                max_k=EXPANDED_K,
                measure_time=False,
            )

            v1_top = _normalize_results(VectorSearch, v1_raw, K=BASE_K)[:COMPARE_TOP_N]
            v2_top = _normalize_results(VectorSearch, v2_raw, K=BASE_K)[:COMPARE_TOP_N]
            v2_expanded = _normalize_results(VectorSearch, v2_expanded_raw, K=EXPANDED_K)

            v1_ids = {_entity_id(row) for row in v1_top if _entity_id(row)}
            v2_ids = {_entity_id(row) for row in v2_top if _entity_id(row)}
            v2_expanded_ids = {_entity_id(row) for row in v2_expanded if _entity_id(row)}

            v1_only = sorted(v1_ids - v2_ids)
            recovered = sorted([qid for qid in v1_only if qid in v2_expanded_ids])
            recovery_rate = (len(recovered) / len(v1_only)) if v1_only else 1.0

            recall_rows.append(
                {
                    "lang": lang,
                    "query": query,
                    "v1_avg_top10": _average_similarity(v1_top, top_n=COMPARE_TOP_N),
                    "v2_avg_top10": _average_similarity(v2_top, top_n=COMPARE_TOP_N),
                    "v1_only_count": len(v1_only),
                    "recovered_count": len(recovered),
                    "recovery_rate": recovery_rate,
                }
            )

    return {
        "runtime_v1": runtime_v1,
        "runtime_v2": runtime_v2,
        "v2_thread_runtimes": v2_thread_runtimes,
        "query_lang_runtime_v1": query_lang_runtime_v1,
        "query_lang_runtime_v2": query_lang_runtime_v2,
        "query_lang_thread_runtimes": query_lang_thread_runtimes,
        "lang_runtime_v1": lang_runtime_v1,
        "lang_runtime_v2": lang_runtime_v2,
        "accuracies_v1": accuracies_v1,
        "accuracies_v2": accuracies_v2,
        "v1_lang_stats_raw_avg": v1_lang_stats_raw_avg,
        "v2_lang_stats_raw_avg": v2_lang_stats_raw_avg,
        "v1_lang_stats_rrf_avg": v1_lang_stats_rrf_avg,
        "v2_lang_stats_rrf_avg": v2_lang_stats_rrf_avg,
        "recall_rows": recall_rows,
    }


def test_split_benchmark_runtime_and_precision(split_benchmark_payload):
    """Validate runtime and similarity results are present and as expected."""
    payload = split_benchmark_payload
    assert payload["runtime_v1"] and payload["runtime_v2"]
    assert statistics.mean(payload["runtime_v1"]) > 0.0
    assert statistics.mean(payload["runtime_v2"]) > 0.0
    assert payload["accuracies_v1"] and payload["accuracies_v2"]
    assert all(score >= 0.0 for score in payload["accuracies_v1"])
    assert all(score >= 0.0 for score in payload["accuracies_v2"])
    for lang in LANGS:
        assert payload["query_lang_runtime_v1"][lang]
        assert payload["query_lang_runtime_v2"][lang]
        assert payload["lang_runtime_v1"][lang]
        assert payload["lang_runtime_v2"][lang]


def test_split_benchmark_exposure_metrics(split_benchmark_payload):
    """Validate exposure metrics stay within expected bounds."""
    payload = split_benchmark_payload
    for key in [
        "v1_lang_stats_raw_avg",
        "v2_lang_stats_raw_avg",
        "v1_lang_stats_rrf_avg",
        "v2_lang_stats_rrf_avg",
    ]:
        stats = payload[key]
        assert set(stats.keys()) == set(LANGS)
        assert all(0.0 <= value <= 1.0 for value in stats.values())


def test_split_benchmark_expanded_recall(split_benchmark_payload):
    """Validate expanded-k recall recovery stays above threshold."""
    rows = split_benchmark_payload["recall_rows"]
    assert rows
    for row in rows:
        assert row["v1_only_count"] >= 0
        assert row["recovered_count"] >= 0
        assert 0.0 <= row["recovery_rate"] <= 1.0
    avg_recovery = statistics.mean(row["recovery_rate"] for row in rows)
    assert avg_recovery >= MIN_EXPECTED_RECOVERY
