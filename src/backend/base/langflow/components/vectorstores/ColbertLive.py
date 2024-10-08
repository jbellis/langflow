from loguru import logger
from typing import Any, List
import torch
from uuid import UUID

from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from langflow.helpers.data import docs_to_data
from langflow.io import (
    DataInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    MultilineInput,
    SecretStrInput,
)
from langflow.schema import Data

from colbert_live.colbert_live import ColbertLive
from colbert_live.models import ColpaliModel
from colbert_live.db import CmdlineDB


class ColbertLiveVectorStoreComponent(LCVectorStoreComponent):
    display_name: str = "ColbertLive"
    description: str = "Implementation of Vector Store using ColbertLive with search capabilities"
    documentation: str = "https://github.com/stanford-futuredata/ColBERT"
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
    ]

    @check_cached_vector_store
    def build_vector_store(self):
        try:
            model = ColpaliModel()
            db = CmdlineDB(self.keyspace, model.dim, UUID(self.api_endpoint), self.token)
            colbert_live = ColbertLive(db, model)
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
                    page_images = [doc.page_content]  # Assuming page_content is an image
                    pngs = []
                    all_embeddings = []
                    for image in page_images:
                        page_embeddings = colbert_live.encode_chunks([image])[0]
                        all_embeddings.append(page_embeddings)
                        pngs.append(image.tobytes())
                    colbert_live.db.add_record(pngs, all_embeddings)
            except Exception as e:
                msg = f"Error adding documents to ColbertLive: {e}"
                raise ValueError(msg) from e
        else:
            logger.debug("No documents to add to the Vector Store.")

    def search_documents(self) -> List[Data]:
        colbert_live = self.build_vector_store()

        logger.debug(f"Search input: {self.search_input}")
        logger.debug(f"Number of results: {self.number_of_results}")

        if self.search_input and isinstance(self.search_input, str) and self.search_input.strip():
            try:
                results = colbert_live.search(self.search_input, k=self.number_of_results)
            except Exception as e:
                msg = f"Error performing search in ColbertLive: {e}"
                raise ValueError(msg) from e

            logger.debug(f"Retrieved documents: {len(results)}")

            # Convert results to Data objects
            data = []
            for chunk_pk, score in results:
                page_body = colbert_live.db.get_page_body(chunk_pk)
                data.append(Data(content=page_body, metadata={"score": score, "chunk_pk": chunk_pk}))

            self.status = data
            return data
        logger.debug("No search input provided. Skipping search.")
        return []

    def get_retriever_kwargs(self):
        return {
            "search_kwargs": {
                "k": self.number_of_results,
            },
        }
