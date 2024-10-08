import uuid
import hashlib
import json
from typing import Any, List, Dict, Optional, Iterable, Tuple
from uuid import UUID

import torch
from cassandra.cluster import ResultSet
from colbert_live.colbert_live import ColbertLive
from colbert_live.db.astra import AstraCQL
from colbert_live.models import ColbertModel
from langchain_core.documents import Document
from langchain.vectorstores.base import VectorStore
from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from langflow.custom.custom_component.component_with_cache import ComponentWithCache
from langflow.io import (
    DataInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    MultilineInput,
    SecretStrInput,
    DictInput,
)
from langflow.schema import Data
from loguru import logger

from base.langflow.inputs.input_mixin import SerializableFieldTypes

class ColbertLiveVectorStore(VectorStore):
    def __init__(self, colbert_live: ColbertLive):
        self.colbert_live = colbert_live

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        bodies = list(texts)
        all_embeddings = self.colbert_live.encode_chunks(bodies)
        
        record_ids = []
        for body, embedding, metadata in zip(bodies, all_embeddings, metadatas):
            record_id = self.colbert_live.db.add_records([body], [embedding], metadata)
            record_ids.append(str(record_id))
        
        return record_ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        query_embedding = self.colbert_live.encode_query(query)
        results = self.colbert_live.db.search_with_metadata_filter(
            query_embedding,
            k,
            kwargs.get("filter", None)
        )
        
        documents = []
        for record_id, score in results:
            body, metadata = self.colbert_live.db.get_record_body(record_id)
            documents.append(Document(page_content=body, metadata={**metadata, "score": score, "record_id": record_id}))
        
        return documents

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        documents = self.similarity_search(query, k, **kwargs)
        return [(doc, doc.metadata["score"]) for doc in documents]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VectorStore:
        raise NotImplementedError("ColbertLiveVectorStore does not support from_texts method")


class ColbertLiveDB(AstraCQL):
    def __init__(self, keyspace: str, table: str, embedding_dim: int, metadata_columns: Dict[str, SerializableFieldTypes], astra_endpoint: str, astra_token: str):
        super().__init__(keyspace, embedding_dim, astra_endpoint=astra_endpoint, astra_token=astra_token, verbose=True)
        self.table = table
        self.metadata_columns = metadata_columns

    def _get_cql_type(self, python_type: type) -> str:
        if python_type == int:
            return "int"
        elif python_type == float:
            return "float"
        elif python_type == bool:
            return "boolean"
        else:
            assert python_type == str
            return "text"

    def prepare(self, embedding_dim: int):
        # Create tables asynchronously
        futures = []

        # Create base table if it doesn't exist
        metadata_columns_str = ", ".join([f"{col_name} {self._get_cql_type(col_type)}" for col_name, col_type in self.metadata_columns.items()])
        futures.append(self.session.execute_async(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.{self.table} (
                record_id uuid PRIMARY KEY,
                body text,
                {metadata_columns_str}
            )
        """))

        # Create embeddings table
        futures.append(self.session.execute_async(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.{self.table}_embeddings (
                record_id uuid,
                embedding_id int,
                embedding vector<float, {embedding_dim}>,
                PRIMARY KEY (record_id, embedding_id)
            )
        """))

        # Wait for all CREATE TABLE operations to complete
        for future in futures:
            future.result()

        # Create colbert_ann index
        index_future = self.session.execute_async(f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {self.table}_ann 
            ON {self.keyspace}.{self.table}_embeddings(embedding) 
            USING 'StorageAttachedIndex'
            WITH OPTIONS = {{ 'source_model': 'bert' }}
        """)

        # Prepare statements
        self.insert_record_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.{self.table} (record_id, body, metadata) VALUES (?, ?, ?)
        """)
        self.insert_embedding_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.{self.table}_embeddings (record_id, embedding_id, embedding) VALUES (?, ?, ?)
        """)

        index_future.result()
        self.query_ann_stmt = self.session.prepare(f"""
            SELECT record_id, similarity_cosine(embedding, ?) AS similarity
            FROM {self.keyspace}.{self.table}_embeddings
            ORDER BY embedding ANN OF ?
            LIMIT ?
        """)
        self.query_chunks_stmt = self.session.prepare(f"""
            SELECT embedding FROM {self.keyspace}.{self.table}_embeddings WHERE record_id = ?
        """)

        print("Schema ready")

    def add_records(self, bodies: List[str], embeddings: List[torch.Tensor], metadata: dict[str, str]):
        record_id = uuid.uuid4()
        L = [(record_id, num, body, metadata) for num, body in enumerate(bodies, start=1)]
        self.session.execute_concurrent_with_args(self.insert_page_stmt, L)

        L = [(record_id, page_num, embedding_id, embedding)
             for page_num in range(1, len(embeddings) + 1)
             for embedding_id, embedding in enumerate(embeddings[page_num - 1])]
        self.session.execute_concurrent_with_args(self.insert_embedding_stmt, L)

        return record_id

    def process_ann_rows(self, result: ResultSet) -> List[tuple[Any, float]]:
        return [(row.record_id, row.similarity) for row in result]

    def process_chunk_rows(self, result: ResultSet) -> List[torch.Tensor]:
        return [torch.tensor(row.embedding) for row in result]

    def get_record_body(self, record_id: Any) -> tuple[str, dict[str, str]]:
        query = f"SELECT body, metadata FROM {self.keyspace}.{self.table} WHERE record_id = %s"
        result = self.session.execute(query, (record_id,))
        row = result.one()
        return row.body, row.metadata

    def search_with_metadata_filter(self, embeddings: torch.Tensor, limit: int, metadata_filter: dict[str, str] = None):
        ann_results = self.query_ann(embeddings, limit * 2)  # Query more results to account for filtering
        filtered_results = []

        for record_id, similarity in ann_results:
            body, metadata = self.get_record_body(record_id)
            if metadata_filter is None or all(metadata.get(k) == v for k, v in metadata_filter.items()):
                filtered_results.append((record_id, similarity))
            if len(filtered_results) == limit:
                break

        return filtered_results[:limit]


class ColbertAstraVectorStoreComponent(LCVectorStoreComponent, ComponentWithCache):
    display_name: str = "ColbertAstra"
    description: str = "An Astra Vector Store using ColbertLive with search capabilities"
    documentation: str = """
    ColbertAstra will infer an appropriate base table schema and create it if ingest_data is specified.
    Otherwise, ColbertAstra assumes that you have created it already.  
    - By convention, the base table must have a single-column primary key named `record_id`, which can be of any type.  
      When ColbertAstra creates the base table, it uses UUID as the PK type.
    - The record body is stored in a column named `body` of type `text`.
      
    ColbertAstra always manages its embeddings table, which is named {base_table}_embeddingshas.  It has three
    fields:
      - record_id
      - embedding_id (int)
      - embedding (vector<float, 96>)
    """
    name = "ColbertLive"
    icon: str = "ColbertLive"

    inputs = [
        SecretStrInput(
            name="token",
            display_name="Astra DB Application Token",
            info="Authentication token for accessing Astra DB.",
            required=True,
        ),
        MessageTextInput(
            name="api_endpoint",
            display_name="API Endpoint",
            info="API endpoint URL for the Astra DB service.",
            required=True,
        ),
        MessageTextInput(
            name="keyspace",
            display_name="Keyspace",
            info="The keyspace to use in the database.",
            required=True,
        ),
        MessageTextInput(
            name="table",
            display_name="Table",
            info="The table to use in the database.",
            required=True,
        ),
        MultilineInput(
            name="search_input",
            display_name="Search Input",
        ),
        DataInput(
            name="ingest_data",
            display_name="Ingest Data",
            is_list=True,
        ),
        HandleInput(
            name="embedding",
            display_name="Embedding Model",
            input_types=["Embeddings"],
            info="Allows an embedding model configuration.",
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            value=4,
        ),
        DictInput(
            name="search_filter",
            display_name="Search Metadata Filter",
            info="Optional dictionary of filters to apply to the search query.",
            advanced=True,
            is_list=True,
        ),
    ]

    def _infer_metadata(self, documents: List[Document]) -> Dict[str, type]:
        metadata_columns = {}
        for doc in documents:
            for key, value in doc.metadata.items():
                if key not in metadata_columns:
                    if isinstance(value, int):
                        metadata_columns[key] = int
                    elif isinstance(value, float):
                        metadata_columns[key] = float
                    elif isinstance(value, bool):
                        metadata_columns[key] = bool
                    elif isinstance(value, str):
                        metadata_columns[key] = str
                    else:
                        raise ValueError(f'Unsupported type {type(value)}')
        return metadata_columns

    def _get_metadata_digest(self, metadata_columns: Dict[str, SerializableFieldTypes]) -> str:
        return hashlib.md5(json.dumps(metadata_columns, sort_keys=True).encode()).hexdigest()

    @check_cached_vector_store
    def build_vector_store(self):
        documents = self._extract_documents()
        metadata_columns = self._infer_metadata(documents)
        metadata_digest = self._get_metadata_digest(metadata_columns)
        cache_key = f"colbert_live_{self.keyspace}_{self.api_endpoint}_{metadata_digest}"
        
        vector_store = self._shared_component_cache.get(cache_key)
        if vector_store is None:
            try:
                model = ColbertModel()
                db = ColbertLiveDB(self.keyspace, self.table, model.dim, metadata_columns, self.api_endpoint, self.token)
                colbert_live = ColbertLive(db, model)
                vector_store = ColbertLiveVectorStore(colbert_live)
                self._shared_component_cache.set(cache_key, vector_store)
            except Exception as e:
                msg = f"Error initializing ColbertLiveVectorStore: {e}"
                raise ValueError(msg) from e

            self._maybe_insert_documents(vector_store, documents)

        return vector_store

    def _maybe_insert_documents(self, vector_store, documents):
        if not documents:
            logger.debug("No documents to add to the Vector Store.")
            return

        logger.debug(f"Adding {len(documents)} documents to the Vector Store.")
        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            vector_store.add_texts(texts, metadatas)

            logger.debug(f"Successfully added {len(documents)} documents to the Vector Store.")
        except Exception as e:
            msg = f"Error adding documents to ColbertLiveVectorStore: {e}"
            raise ValueError(msg) from e

    def _extract_documents(self):
        documents = []
        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                msg = "Vector Store Inputs must be Data objects."
                raise ValueError(msg)
        return documents

    def search_documents(self) -> List[Data]:
        colbert_live = self.build_vector_store()

        logger.debug(f"Search input: {self.search_input}")
        logger.debug(f"Number of results: {self.number_of_results}")
        logger.debug(f"Search filter: {self.search_filter}")

        if self.search_input and isinstance(self.search_input, str) and self.search_input.strip():
            try:
                query_embedding = colbert_live.encode_query(self.search_input)
                results = colbert_live.db.search_with_metadata_filter(
                    query_embedding,
                    self.number_of_results,
                    self.search_filter
                )
            except Exception as e:
                msg = f"Error performing search in ColbertLive: {e}"
                raise ValueError(msg) from e

            logger.debug(f"Retrieved documents: {len(results)}")

            # Convert results to Data objects
            data = []
            for record_id, score in results:
                body, metadata = colbert_live.db.get_record_body(record_id)
                data.append(Data(content=body, metadata={**metadata, "score": score, "record_id": record_id}))

            self.status = data
            return data
        logger.debug("No search input provided. Skipping search.")
        return []

    def get_retriever_kwargs(self):
        return {
            "search_kwargs": {
                "k": self.number_of_results,
                "filter": self.search_filter,
            },
        }
