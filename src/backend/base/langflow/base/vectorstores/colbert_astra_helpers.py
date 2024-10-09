import uuid
from typing import Any, List, Dict, Optional, Iterable, Tuple

import torch
from cassandra.cluster import ResultSet
from colbert_live.colbert_live import ColbertLive
from colbert_live.db.astra import AstraCQL
from langchain.vectorstores.base import VectorStore
from langchain_core.documents import Document


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

    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        query_embedding = self.colbert_live.encode_query(query)
        results = self.colbert_live.search(query_embedding, k, filter=filter or {})

        documents = []
        for record_id, score in results:
            body, metadata = self.colbert_live.db.get_record_body(record_id)
            documents.append(Document(page_content=body, metadata={**metadata, "score": score, "record_id": record_id}))

        return documents

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
    def __init__(self, keyspace: str, table: str, embedding_dim: int, metadata_columns: List[Tuple[str, type]],
                 astra_endpoint: str, astra_token: str):
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
        metadata_columns_create = ", ".join(
            [f"{col_name} {self._get_cql_type(col_type)}" for col_name, col_type in self.metadata_columns])
        futures.append(self.session.execute_async(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.{self.table} (
                record_id uuid PRIMARY KEY,
                body text,
                {metadata_columns_create}
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
        metadata_insert_columns = ", ".join([col_name for col_name, _ in self.metadata_columns])
        metadata_insert_binds = ", ".join(["?"] * len(self.metadata_columns))
        self.insert_record_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.{self.table} (record_id, body, {metadata_insert_columns}) 
            VALUES (?, ?, {metadata_insert_binds})
        """)
        self.insert_embedding_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.{self.table}_embeddings (record_id, embedding_id, embedding) VALUES (?, ?, ?)
        """)

        index_future.result()
        metadata_select_where = " AND ".join([f"metadata.{col_name} = ?" for col_name, _ in self.metadata_columns])
        self.query_ann_stmt = self.session.prepare(f"""
            SELECT record_id, similarity_cosine(embedding, ?) AS similarity
            FROM {self.keyspace}.{self.table}_embeddings
            {metadata_select_where}
            ORDER BY embedding ANN OF ?
            LIMIT ?
        """)
        self.query_chunks_stmt = self.session.prepare(f"""
            SELECT embedding FROM {self.keyspace}.{self.table}_embeddings WHERE record_id = ?
        """)

        print("Schema ready")

    def add_records(self, bodies: List[str], embeddings: List[torch.Tensor], metadata: List[Dict[str, str]]):
        record_id = uuid.uuid4()
        record_vars = [(record_id, body, *[metadata.get(col_name, None) for col_name, _ in self.metadata_columns])
                       for body, metadata in zip(bodies, metadata)]
        self.session.execute_concurrent_with_args(self.insert_record_stmt, record_vars)

        embedding_vars = [(record_id, i, embedding)
                          for i, embedding in enumerate(embeddings)]
        self.session.execute_concurrent_with_args(self.insert_embedding_stmt, embedding_vars)

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
