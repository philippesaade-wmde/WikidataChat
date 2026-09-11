"""Vector search implementation backed by Astra DB collections."""

import re

from astrapy import DataAPIClient
from astrapy.api_options import APIOptions, TimeoutOptions

from ..jina import JinaAIAPI
from .Search import Search


class VectorSearch(Search):
    """Search implementation that uses vector similarity over stored embeddings."""

    name = "Vector Search"

    def __init__(self, api_keys, collection: str, id_field: str = "QID", embedding_model=None, max_K: int = 50):
        """Initialize the Vector Database connection and embedding model.

        Args:
            api_keys (dict): API credentials for AstraDB and Jina.
            collection (str): Exact collection name queried by this instance.
            id_field (str): The field name used for entity IDs. Defaults to 'QID'.
            embedding_model (object, optional): Pre-initialized embedding model.
            max_K (int, optional): Maximum nearest-neighbor result size.
        """
        ASTRA_DB_APPLICATION_TOKEN = api_keys["ASTRA_DB_APPLICATION_TOKEN"]
        ASTRA_DB_API_ENDPOINT = api_keys["ASTRA_DB_API_ENDPOINT"]
        JINA_API_KEY = api_keys["JINA_API_KEY"]

        timeout_options = TimeoutOptions(request_timeout_ms=100000)
        api_options = APIOptions(timeout_options=timeout_options)

        client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN, api_options=api_options)
        database0 = client.get_database(ASTRA_DB_API_ENDPOINT)
        self.collection = database0.get_collection(collection)
        self.id_field = id_field

        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = JinaAIAPI(JINA_API_KEY)

        self.max_K = max_K

    def search(
        self,
        query: str,
        filter: dict | None = None,
        embedding: list | None = None,
        lang: str = "all",
        K: int = 50,
        return_vectors: bool = False,
        return_text: bool = False,
        exact_match: dict | None = None,
    ) -> list:
        """Retrieve similar Wikidata entities from the vector database for a given query string.

        Args:
            query (str): The search query string.
            filter (dict, optional): Additional filtering criteria.
            embedding (list | None, optional): Precomputed query embedding.
            lang (str): The language of the vectors to query. Defaults to 'all'.
            K (int, optional): Number of top results to return. Defaults to 50.
            return_vectors (bool): Whether to include vectors in the response.
            return_text (bool): Whether to include text content in the response.
            exact_match (dict | None): Entity used to resolve a supplied ID embedding.

        Returns:
            list: Deduplicated entities with QID/PID and similarity scores.
        """
        query_filter = dict(filter or {})
        relevant_items = []
        item = exact_match

        if embedding is None:
            embedding, item = self.calculate_embedding(query, lang=lang, return_text=return_text)

        if item:
            item["$similarity"] = 1.0
            relevant_items.append(item)

        if embedding is None:
            return relevant_items

        projection = {"metadata": 1}
        if return_text:
            projection["content"] = 1
        if return_vectors:
            projection["$vector"] = 1

        relevant_items.extend(
            self.find(
                query_filter,
                sort={"$vector": embedding},
                projection=projection,
                limit=K,
                include_similarity=True,
            )
        )

        relevant_items = VectorSearch.remove_duplicates(
            relevant_items, return_vectors=return_vectors, return_text=return_text
        )
        return relevant_items

    def calculate_embedding(self, query, lang: str = "en", return_text=False):
        """Compute query embedding, resolving entity IDs to stored vectors when possible.

        Args:
            query: Natural-language query or Wikidata entity ID.
            lang: Language used to resolve fallback text for entity IDs.
            return_text: Whether downstream calls should also fetch text payloads.

        Returns:
            tuple[list | None, dict | None]: Query embedding and exact-match entity.
        """
        if re.fullmatch(r"[PQ]\d+", query):
            item, embedding = self.get_embedding_by_id(
                query,
                return_text=return_text,
            )

            if not item:
                try:
                    query_text = self.get_text_by_ids([query], format="text", lang=lang)

                    query = query_text.get(query)
                    if not query:
                        return None, None
                except Exception:
                    return None, None
            else:
                return embedding, item

        embedding = self.embedding_model.embed_query(query)
        return embedding, None

    def get_similarity_scores(
        self,
        query: str,
        ids: list,
        embedding: list | None = None,
        lang: str = "all",
        return_vectors: bool = False,
        return_text: bool = False,
    ) -> list:
        """Retrieve similarity scores for specific Wikidata IDs using one query.

        Args:
            query (str): The search query string.
            ids (list): Wikidata IDs to retrieve from this instance's collection.
            embedding (list | None, optional): Precomputed query embedding.
            lang (str): The language of the vectors to query. Defaults to 'all'.
            return_vectors (bool): Whether to return the vector embeddings of the entity.
            return_text (bool): Whether to return the text content of the entity.

        Returns:
            list: Matching entities with similarity scores.
        """
        if not ids:
            return []

        if len(ids) > 100:
            raise ValueError("Too many IDs provided for similarity scoring. Please provide 100 or fewer IDs.")

        if embedding is None:
            embedding, _ = self.calculate_embedding(query, lang=lang, return_text=return_text)

        if embedding is None:
            return []

        ids = list(set(ids))

        projection = {
            "metadata": 1,
            "$vector": 1,
        }
        if return_text:
            projection["content"] = 1

        query_filter = {f"metadata.{self.id_field}": {"$in": ids}}
        results = self.find(
            query_filter,
            projection=projection,
            limit=None,
        )

        relevant_items = []
        for item in results:
            similarity = self.embedding_model.similarity(embedding, item.get("$vector"))
            relevant_items.append({**item, "$similarity": similarity})

        relevant_items = VectorSearch.remove_duplicates(
            relevant_items, return_vectors=return_vectors, return_text=return_text
        )
        return relevant_items

    def get_embedding_by_id(self, qid, return_text=False):
        """Fetch the stored embedding for one Wikidata entity ID.

        Args:
            qid (str): A Wikidata entity ID (QID or PID).
            return_text (bool): Whether to return the text content of the entity.

        Returns:
            tuple[dict, list | None]: The matching database record and its vector.
        """
        filter = {f"metadata.{self.id_field}": qid}

        projection = {"metadata": 1, "$vector": 1}
        if return_text:
            projection["content"] = 1

        results = self.find(
            filter,
            projection=projection,
            limit=1,
        )

        item = results[0] if results else {}
        return item, item.get("$vector")

    def find(self, filter, sort=None, projection=None, limit=50, include_similarity=True):
        """Run a low-level Astra DB query against this instance's collection.

        Args:
            filter: Astra DB filter expression.
            sort: Optional sort clause, including vector sort.
            projection: Optional field projection.
            limit: Maximum number of rows to return when sorting.
            include_similarity: Whether Astra should include similarity values.

        Returns:
            list: Raw Astra records matching the query.
        """
        query_filter = dict(filter or {})

        if sort:
            if limit is None:
                limit = self.max_K
            limit = max(1, min(limit, self.max_K))

            results = self.collection.find(
                query_filter,
                sort=sort,
                projection=projection or {"metadata": 1},
                limit=limit,
                include_similarity=include_similarity,
            )
        else:
            results = self.collection.find(
                query_filter,
                projection=projection or {"metadata": 1},
            )
        return list(results)

    @staticmethod
    def remove_duplicates(results, return_vectors=False, return_text=False):
        """Collapse duplicate entities and keep the highest-scoring representation.

        Args:
            results: Raw vector-search records to normalize.
            return_vectors: Whether to keep vector values in output rows.
            return_text: Whether to keep text values in output rows.

        Returns:
            list: Deduplicated dictionaries keyed by QID or PID.
        """
        results = sorted(results, key=lambda x: x.get("$similarity", x.get("similarity_score", 0.0)), reverse=True)

        seen_qids = set()
        output = []

        for item in results:
            metadata = item.get("metadata", {})
            ID = metadata.get("QID", metadata.get("PID"))
            if not ID:
                continue
            if ID not in seen_qids:
                ID_name = "QID" if ID.startswith("Q") else "PID"
                item_output = {
                    ID_name: ID,
                    "similarity_score": item.get("$similarity", item.get("similarity_score", 0.0)),
                }
                if return_vectors:
                    item_output["vector"] = item.get("$vector")
                if return_text:
                    item_output["text"] = item.get("content", item.get("text"))

                output.append(item_output)

                seen_qids.add(ID)

        return output
