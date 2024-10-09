import hashlib
import json
from typing import List, Tuple

from colbert_live.colbert_live import ColbertLive
from colbert_live.models import ColbertModel
from langchain_core.documents import Document
from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
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
from base.langflow.services.deps import get_shared_component_cache_service

from base.langflow.base.vectorstores.colbert_astra_helpers import ColbertLiveDB, ColbertLiveVectorStore


class ColbertAstraVectorStoreComponent(LCVectorStoreComponent):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._shared_component_cache = get_shared_component_cache_service()

    def _infer_metadata(self, documents: List[Document]) -> List[Tuple[str, type]]:
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
        return metadata_columns.items().sorted()

    def _get_metadata_digest(self, metadata_columns: List[Tuple[str, SerializableFieldTypes]]) -> str:
        return hashlib.md5(json.dumps(metadata_columns).encode()).hexdigest()

    @check_cached_vector_store
    def build_vector_store(self):
        documents = self._extract_documents()
        metadata_columns = self._infer_metadata(documents)
        metadata_digest = self._get_metadata_digest(metadata_columns)
        cache_key = f"colbert_live_{self.keyspace}_{self.api_endpoint}_{metadata_digest}"
        
        vector_store = self._shared_component_cache.get(cache_key)
        if vector_store is None:
            model = ColbertModel()
            db = ColbertLiveDB(self.keyspace, self.table, model.dim, metadata_columns, self.api_endpoint, self.token)
            colbert_live = ColbertLive(db, model)
            vector_store = ColbertLiveVectorStore(colbert_live)
            self._shared_component_cache.set(cache_key, vector_store)
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
