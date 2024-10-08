import uuid
from typing import Any, List
from uuid import UUID

import torch
from cassandra.cluster import ResultSet
from colbert_live.colbert_live import ColbertLive
from colbert_live.db.astra import AstraCQL
from colbert_live.models import ColbertModel
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


class ColbertLiveDB(AstraCQL):
    def __init__(self, keyspace: str, table: str, embedding_dim: int, astra_db_id: str, astra_token: str):
        super().__init__(keyspace, embedding_dim, astra_db_id, astra_token, verbose=True)
        self.table = table

    def prepare(self, embedding_dim: int):
        # Create tables asynchronously
        futures = []

        # Create base table if it doesn't exist
        futures.append(self.session.execute_async(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.{self.table} (
                record_id uuid PRIMARY KEY,
                content text,
                metadata map<text, text>
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
            INSERT INTO {self.keyspace}.{self.table} (record_id, content, metadata) VALUES (?, ?, ?)
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

    def add_record(self, pages: list[bytes], embeddings: list[torch.Tensor], metadata: dict[str, str]):
        record_id = uuid.uuid4()
        L = [(record_id, num, body, metadata) for num, body in enumerate(pages, start=1)]
        self.session.execute_concurrent_with_args(self.insert_page_stmt, L)

        L = [(record_id, page_num, embedding_id, embedding)
             for page_num in range(1, len(embeddings) + 1)
             for embedding_id, embedding in enumerate(embeddings[page_num - 1])]
        self.session.execute_concurrent_with_args(self.insert_embedding_stmt, L)

        return record_id

    def process_ann_rows(self, result: ResultSet) -> List[tuple[Any, float]]:
        return [((row.record_id, row.page_num), row.similarity) for row in result]

    def process_chunk_rows(self, result: ResultSet) -> List[torch.Tensor]:
        return [torch.tensor(row.embedding) for row in result]

    def get_page_body(self, chunk_pk: tuple) -> bytes:
        record_id, page_num = chunk_pk
        query = f"SELECT body, metadata FROM {self.keyspace}.pages WHERE record_id = %s AND num = %s"
        result = self.session.execute(query, (record_id, page_num))
        row = result.one()
        return row.body, row.metadata

    def search_with_metadata_filter(self, embeddings: torch.Tensor, limit: int, metadata_filter: dict[str, str] = None):
        ann_results = self.query_ann(embeddings, limit * 2)  # Query more results to account for filtering
        filtered_results = []

        for chunk_results in ann_results:
            for chunk_pk, similarity in chunk_results:
                _, metadata = self.get_page_body(chunk_pk)
                if metadata_filter is None or all(metadata.get(k) == v for k, v in metadata_filter.items()):
                    filtered_results.append((chunk_pk, similarity))
                if len(filtered_results) == limit:
                    break
            if len(filtered_results) == limit:
                break

        return filtered_results[:limit]


class ColbertAstraVectorStoreComponent(LCVectorStoreComponent, ComponentWithCache):
    display_name: str = "ColbertAstra"
    description: str = "An Astra Vector Store using ColbertLive with search capabilities"
    documentation: str = """
    ColbertAstra will infer an appropriate base table schema and create it if ingest_data is specified.
    Otherwise, ColbertAstra assumes that you have created it already.  By convention, the base table
    must have a single-column primary key named `record_id`, which can be of any type.  When ColbertAstra
    creates the base table, it uses UUID as the PK type.
      
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

    @check_cached_vector_store
    def build_vector_store(self):
        cache_key = f"colbert_live_{self.keyspace}_{self.api_endpoint}"
        
        colbert_live = self._shared_component_cache.get(cache_key)
        if colbert_live is None:
            try:
                model = ColbertModel()
                db = ColbertLiveDB(self.keyspace, model.dim, UUID(self.api_endpoint), self.token)
                colbert_live = ColbertLive(db, model)
                self._shared_component_cache.set(cache_key, colbert_live)
            except Exception as e:
                msg = f"Error initializing ColbertLive: {e}"
                raise ValueError(msg) from e

        self._add_documents_to_vector_store(colbert_live)

        return colbert_live

    def _add_documents_to_vector_store(self, colbert_live):
        documents = []
        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                msg = "Vector Store Inputs must be Data objects."
                raise ValueError(msg)

        if documents:
            logger.debug(f"Adding {len(documents)} documents to the Vector Store.")
            try:
                for doc in documents:
                    content = doc.page_content
                    metadata = doc.metadata
                    
                    # Generate embeddings for the document
                    embeddings = colbert_live.encode_chunks([content])
                    
                    # Add the document and its embeddings to the database
                    record_id = uuid.uuid4()
                    colbert_live.db.session.execute(colbert_live.db.insert_record_stmt, (record_id, content, metadata))
                    
                    for embedding_id, embedding in enumerate(embeddings[0]):
                        colbert_live.db.session.execute(colbert_live.db.insert_embedding_stmt, (record_id, embedding_id, embedding.tolist()))
                    
                logger.debug(f"Successfully added {len(documents)} documents to the Vector Store.")
            except Exception as e:
                msg = f"Error adding documents to ColbertLive: {e}"
                raise ValueError(msg) from e
        else:
            logger.debug("No documents to add to the Vector Store.")

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
            for chunk_pk, score in results:
                page_body, metadata = colbert_live.db.get_page_body(chunk_pk)
                data.append(Data(content=page_body, metadata={**metadata, "score": score, "chunk_pk": chunk_pk}))

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
