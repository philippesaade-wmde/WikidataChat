"""Hybrid search combining vector, keyword, translation, and reranking flows."""

import re
import traceback
from concurrent.futures import ThreadPoolExecutor

from ..jina import JinaAIAPI
from ..translator import Translator
from .KeywordSearch import KeywordSearch
from .Search import Search
from .VectorSearch import VectorSearch


class HybridSearch(Search):
    """Search implementation that combines vector and keyword results."""

    name = "Hybrid Search"

    def __init__(self, api_keys, dest_lang: str = "en", vectordb_langs: list[str] | None = None, max_K: int = 50):
        """Initialize hybrid search with keyword and per-language vector backends.

        Args:
            api_keys (dict): API credentials and AstraDB configuration.
            dest_lang (str, optional): Default translation target language.
            vectordb_langs (list[str] | None, optional): Languages available in the vector database.
            max_K (int, optional): Maximum number of vector neighbors requested per shard.
        """
        vectordb_langs = vectordb_langs or []
        collection = api_keys["ASTRA_DB_COLLECTION"]
        self.embedding_model = JinaAIAPI(api_keys["JINA_API_KEY"])
        self.vectordb_langs = vectordb_langs

        self.vectorsearch = {
            "items": {
                lang: VectorSearch(
                    api_keys,
                    f"{collection}_items_{lang}",
                    id_field="QID",
                    embedding_model=self.embedding_model,
                    max_K=max_K,
                )
                for lang in vectordb_langs
            },
            "properties": {
                lang: VectorSearch(
                    api_keys,
                    f"{collection}_properties_{lang}",
                    id_field="PID",
                    embedding_model=self.embedding_model,
                    max_K=max_K,
                )
                for lang in vectordb_langs
            },
        }
        self.keywordsearch = KeywordSearch()
        self.translator = Translator(dest_lang)

    def _search_vector_shard(
        self,
        vectorsearch: VectorSearch,
        query: str,
        query_filter: dict,
        embedding: list | None,
        lang: str,
        K: int,
        return_vectors: bool,
    ) -> list:
        """Search one target shard, resolving ID embeddings from the matching entity collection."""
        exact_match = None
        shard_embedding = embedding

        if shard_embedding is None and re.fullmatch(r"[PQ]\d+", query):
            entity_type = "properties" if query.startswith("P") else "items"
            embedding_search = self.vectorsearch[entity_type][lang]
            shard_embedding, resolved_match = embedding_search.calculate_embedding(query, lang=lang)
            if shard_embedding is None:
                return []
            if query.startswith(vectorsearch.id_field[0]):
                exact_match = resolved_match

        return vectorsearch.search(
            query,
            filter=query_filter,
            embedding=shard_embedding,
            lang=lang,
            K=K,
            return_vectors=return_vectors,
            exact_match=exact_match,
        )

    def search(
        self,
        query: str,
        filter: dict | None = None,
        embedding: list | None = None,
        vs_K: int = 50,
        ks_K: int = 5,
        lang: str = "all",
        rerank: bool = False,
        return_vectors: bool = False,
    ) -> list:
        """Search for items based on the query and filter using both keyword and vector search.

        Args:
            query (str): The search query string.
            filter (dict, optional): Additional filtering criteria.
            embedding (list | None, optional): Precomputed query embedding.
            vs_K (int, optional): Number of top results from Vector Search. Defaults to 50.
            ks_K (int, optional): Number of top results from Keyword Search. Defaults to 5.
            lang (str): The source language of the query. Defaults to "all".
            rerank (bool): Whether to rerank the results. Defaults to False.
            return_vectors (bool): Whether to return the vector embeddings of the entity. Defaults to False.

        Returns:
            list: A list of fused search results for items and/or properties.
        """
        query_filter = dict(filter or {})
        is_id = re.fullmatch(r"[PQ]\d+", query)
        if query_filter.get("metadata.IsProperty", False):
            vectorsearch = self.vectorsearch["properties"]
        elif query_filter.get("metadata.IsItem", False):
            vectorsearch = self.vectorsearch["items"]
        elif "metadata.PID" in query_filter:
            vectorsearch = self.vectorsearch["properties"]
        else:
            vectorsearch = self.vectorsearch["items"]

        vector_filter = query_filter.copy()
        vector_filter.pop("metadata.IsItem", None)
        vector_filter.pop("metadata.IsProperty", None)

        lang = (lang or "all").lower()
        vector_query = query

        if lang != "all" and lang not in vectorsearch:
            # Translate only if we are about to compute embedding here
            if not is_id and embedding is None:
                vector_query = self.translator.translate(query, src_lang=lang)
            lang = "all"

        # Reuse embedding when provided
        if not is_id and embedding is None:
            embedding = self.embedding_model.embed_query(vector_query)

        num_shards = sum([int(vdblang == lang or lang == "all") for vdblang, _ in vectorsearch.items()])
        num_shards = max(num_shards, 1)
        vs_K = max(10, min(vs_K, (vs_K * 2 + 1) // num_shards))

        with ThreadPoolExecutor(max_workers=4) as ex:
            vfunc = []
            for vdblang, vdb in vectorsearch.items():
                if vdblang == lang or lang == "all":
                    func = ex.submit(
                        self._search_vector_shard,
                        vdb,
                        vector_query,
                        vector_filter.copy(),
                        embedding,
                        vdblang,
                        vs_K,
                        return_vectors,
                    )
                    vfunc.append((vdblang, func))

            kfunc = ex.submit(
                self.keyword_search,
                query,
                filter=query_filter.copy(),
                embedding=embedding,
                lang=lang,
                K=ks_K,
                return_vectors=return_vectors,
            )

            vector_results = {vdblang: f.result() for vdblang, f in vfunc}
            keyword_results = kfunc.result()

        # Combine results using Reciprocal Rank Fusion
        combined_results = [(vectorsearch[vdblang].name, vector_results[vdblang]) for vdblang, _ in vfunc]
        combined_results.append((self.keywordsearch.name, keyword_results))
        results = self.reciprocal_rank_fusion(combined_results)
        results = results[:vs_K]

        if rerank:
            # Rerank the results with the current Wikidata values.
            ids = [r.get("QID", r.get("PID")) for r in results]
            ids = [rid for rid in ids if rid]
            if not ids:
                return results

            wd_data = self.get_text_by_ids(ids, format="triplet", lang=lang)
            for i in range(len(results)):
                rid = results[i].get("QID", results[i].get("PID"))
                if rid in wd_data:
                    results[i]["text"] = wd_data[rid]

            results = [r for r in results if r.get("text")]
            if not results:
                return results

            results = self.embedding_model.rerank(query, results)

            # Remove text from results to reduce payload size
            for r in results:
                r.pop("text", None)

        return results

    def keyword_search(
        self,
        query: str,
        filter: dict | None = None,
        embedding: list | None = None,
        lang: str = "all",
        K: int = 50,
        return_vectors: bool = False,
        return_text: bool = False,
    ) -> list:
        """Run keyword search and score keyword hits against the query embedding.

        Args:
            query (str): The query string.
            filter (dict | None, optional): Filters forwarded to keyword search.
            embedding (list | None, optional): Optional precomputed query embedding.
            lang (str, optional): Query language code. Defaults to "all".
            K (int, optional): Maximum number of keyword candidates. Defaults to 50.
            return_vectors (bool, optional): Include vectors in returned results.
            return_text (bool, optional): Include text fields in returned results.

        Returns:
            list: Scored keyword results.
        """
        filter = filter or {}

        # Perform keyword search
        try:
            keyword_results = self.keywordsearch.search(query, filter=filter, lang=lang, K=K)
        except Exception:
            traceback.print_exc()
            return []

        # Get similarity scores for keyword results
        keyword_results = self.get_similarity_scores(
            query,
            keyword_results,
            embedding=embedding,
            lang=lang,
            return_vectors=return_vectors,
            return_text=return_text,
        )

        return keyword_results

    def get_similarity_scores(
        self,
        query: str,
        qids: list,
        embedding: list | None = None,
        lang: str = "all",
        return_vectors: bool = False,
        return_text: bool = False,
    ) -> list:
        """Get similarity scores for a list of items against a query.

        Args:
            query (str): The query string.
            qids (list): The list of Wikidata IDs (QIDs/PIDs) to compare against.
            embedding (list | None, optional): Optional precomputed query embedding.
            lang (str): Query language. Defaults to "all".
            return_vectors (bool): Whether to return vector representations.
            return_text (bool): Whether to return text representations.

        Returns:
            list: Similarity-scored entities sorted in descending score order.
        """
        if not qids:
            return []

        if len(qids) > 100:
            raise ValueError("Too many QIDs provided for similarity scoring. Please provide 100 or fewer QIDs.")

        is_id = re.fullmatch(r"[PQ]\d+", query)
        lang = (lang or "all").lower()
        vector_query = query

        if lang != "all" and lang not in self.vectorsearch["items"]:
            # Translate only if we are about to compute embedding here
            if not is_id and embedding is None:
                vector_query = self.translator.translate(query, src_lang=lang)
            lang = "all"

        # Reuse embedding when provided
        if not is_id and embedding is None:
            embedding = self.embedding_model.embed_query(vector_query)

        qids = list(set(qids))
        q_list = [qid for qid in qids if qid.startswith("Q")]
        p_list = [pid for pid in qids if pid.startswith("P")]

        with ThreadPoolExecutor(max_workers=4) as ex:
            vfunc = []
            for vdblang in self.vectorsearch["items"]:
                if vdblang == lang or lang == "all":
                    func = ex.submit(
                        self._get_similarity_scores_for_shard,
                        vector_query,
                        q_list,
                        p_list,
                        embedding,
                        vdblang,
                        return_vectors,
                        return_text,
                    )
                    vfunc.append((vdblang, func))

            vector_results = [item for _, future in vfunc for item in future.result()]

        best_by_id = {}
        for item in vector_results:
            entity_id = item.get("QID") or item.get("PID")
            if not entity_id:
                continue
            previous = best_by_id.get(entity_id)
            if previous is None or item.get("similarity_score", 0.0) > previous.get("similarity_score", 0.0):
                best_by_id[entity_id] = item

        results = sorted(best_by_id.values(), key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        return results[: len(qids)]

    def _get_similarity_scores_for_shard(
        self,
        query: str,
        qids: list,
        pids: list,
        embedding: list | None,
        lang: str,
        return_vectors: bool,
        return_text: bool,
    ) -> list:
        """Score item and property IDs in their respective collections for one language shard."""
        shard_embedding = embedding
        if shard_embedding is None:
            entity_type = "properties" if query.startswith("P") else "items"
            embedding_search = self.vectorsearch[entity_type][lang]
            shard_embedding, _ = embedding_search.calculate_embedding(
                query,
                lang="all",
                return_text=return_text,
            )

        if shard_embedding is None:
            return []

        results = []
        if qids:
            results.extend(
                self.vectorsearch["items"][lang].get_similarity_scores(
                    query,
                    qids,
                    embedding=shard_embedding,
                    return_vectors=return_vectors,
                    return_text=return_text,
                )
            )
        if pids:
            results.extend(
                self.vectorsearch["properties"][lang].get_similarity_scores(
                    query,
                    pids,
                    embedding=shard_embedding,
                    return_vectors=return_vectors,
                    return_text=return_text,
                )
            )
        return results

    @staticmethod
    def reciprocal_rank_fusion(results: list, k: int = 50) -> list:
        """Combine result lists with Reciprocal Rank Fusion (RRF).

        Args:
            results (list): Sequence of `(source_name, source_results)` tuples.
            k (int): Smoothing factor for rank contribution.

        Returns:
            list[dict]: Fused results including QID/PID, similarity score, source, and `rrf_score`.
        """
        scores = {}

        for source_name, source_results in results:
            for rank, item in enumerate(source_results):
                ID = item.get("QID", item.get("PID"))

                similarity_score = item.get("similarity_score", 0.0)
                rrf_score = 1.0 / (k + rank + 1)

                if similarity_score > 0.0:
                    if ID not in scores:
                        scores[ID] = {
                            **item,
                            "rrf_score": rrf_score,
                            "source": source_name,
                        }

                    else:
                        scores[ID]["similarity_score"] = max(similarity_score, scores[ID].get("similarity_score", 0.0))
                        scores[ID]["rrf_score"] += rrf_score

                        if source_name not in scores[ID]["source"]:
                            scores[ID]["source"] += f", {source_name}"

        fused_results = sorted(
            scores.values(), key=lambda x: (x["rrf_score"], x.get("similarity_score", 0.0)), reverse=True
        )
        return fused_results
