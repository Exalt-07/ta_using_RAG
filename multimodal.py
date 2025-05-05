from PIL import Image
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import numpy as np

class MultimodalEngine:
    def __init__(self):
        self.clip = SentenceTransformer('clip-ViT-B-32')
        self.client = chromadb.PersistentClient(path="data/processed/chroma_images")
        self.collection = self.client.get_or_create_collection("image_embeddings")
        self.image_paths = []

    def process_images(self, image_dir):
        """Process images and store embeddings in ChromaDB"""
        image_paths = [str(f) for f in image_dir.glob('*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        images = [Image.open(p) for p in image_paths]
        
        # Generate CLIP embeddings
        embeddings = self.clip.encode(images).tolist()
        
        # Store in Chroma with image paths as metadata
        self.collection.add(
            embeddings=embeddings,
            metadatas=[{"path": p} for p in image_paths],
            ids=[str(i) for i in range(len(image_paths))]
        )
        self.image_paths = image_paths

    def image_search(self, text_query, top_k=3):
        """Search images using text query"""
        # Convert text to embedding
        text_embed = self.clip.encode(text_query).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[text_embed],
            n_results=top_k
        )
        
        # Return image paths from metadata
        return [Path(m['path']) for m in results['metadatas'][0]]
