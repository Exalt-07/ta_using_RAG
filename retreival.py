# app/retrieval.py
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import logging
import shutil
from pathlib import Path
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.errors import ChromaError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model_name = model_name
        self.text_model = SentenceTransformer(model_name)
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.expected_dim = self.text_model.get_sentence_embedding_dimension()
        self.client = chromadb.PersistentClient(path="data/processed/chromadb")
        self.collection = self._get_valid_collection()
        self.bm25_index = None
        self.documents = []

    def _get_valid_collection(self):
        try:
            collection = self.client.get_collection("lecture_materials")
            
            # Validate collection properties
            if not collection.metadata:
                logger.warning("Legacy collection without metadata found")
                raise ValueError("Invalid metadata")
                
            actual_dim = int(collection.metadata.get("dimension", -1))
            if actual_dim != self.expected_dim:
                logger.warning(f"Dimension mismatch ({actual_dim} vs {self.expected_dim})")
                raise ValueError("Dimension mismatch")
                
            return collection
            
        except (ChromaError, ValueError) as e:
            logger.info(f"Creating new collection: {str(e)}")
            self._cleanup_legacy_data()
            return self.client.create_collection(
                name="lecture_materials",
                embedding_function=self.embedding_function,
                metadata={
                    "dimension": str(self.expected_dim),
                    "model": self.model_name
                }
            )

    def _cleanup_legacy_data(self):
        """Thorough cleanup of existing data"""
        try:
            self.client.delete_collection("lecture_materials")
        except Exception as e:
            logger.info(f"Collection delete error: {str(e)}")
            
        persistence_dir = Path("data/processed/chromadb")
        if persistence_dir.exists():
            for item in persistence_dir.glob("*"):
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    logger.error(f"Cleanup failed for {item}: {str(e)}")

    # Rest of the class remains unchanged

    def build_indices(self, documents):
        """Build both vector and keyword search indices"""
        if not documents:
            logger.warning("No documents provided for indexing")
            return

        self.documents = documents
        
        try:
            # Generate and store embeddings
            embeddings = self.text_model.encode(documents).tolist()
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=[str(i) for i in range(len(documents))],
                metadatas=[{"source": "document"} for _ in documents]
            )
            
            # Build BM25 index
            tokenized_docs = [doc.split() for doc in documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
            logger.info(f"Built indices for {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error building indices: {str(e)}")
            raise

    def search(self, query, top_k=5):
        """Hybrid search combining vector and keyword results"""
        if not self.documents:
            logger.warning("No documents indexed - returning empty results")
            return []

        try:
            # Vector search
            vector_results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            vector_ids = [int(id) for id in vector_results["ids"][0]]
            
            # BM25 search
            tokenized_query = query.split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            bm25_ids = np.argsort(bm25_scores)[-top_k:][::-1]
            
            # Combine and deduplicate results
            combined_ids = np.unique(np.concatenate([vector_ids, bm25_ids]))
            return [self.documents[i] for i in combined_ids[:top_k]]
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
