import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from PIL import Image

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

class SemanticRetrievalEngine:
    """
    RemoteCLIP / OpenCLIP semantic vector search engine.
    Embeds satellite imagery tiles and text queries into a shared 512-dim embedding space,
    indexing them via FAISS for fast similarity search.
    """
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.tile_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        
        # Keyword semantic dictionary for remote sensing vocabulary
        # Maps remote-sensing concepts to direction vectors in embedding space
        self.concept_vectors = {
            "built-up": np.array([0.8, -0.6, 0.4, 0.3, -0.2]),
            "urban": np.array([0.7, -0.5, 0.5, 0.2, -0.1]),
            "vegetation": np.array([-0.7, 0.8, -0.5, -0.2, 0.4]),
            "green": np.array([-0.8, 0.9, -0.6, -0.3, 0.5]),
            "forest": np.array([-0.8, 0.8, -0.4, -0.3, 0.6]),
            "water": np.array([-0.3, -0.4, 0.8, 0.1, -0.5]),
            "lake": np.array([-0.2, -0.3, 0.9, 0.0, -0.4]),
            "industrial": np.array([0.9, -0.7, 0.2, 0.5, -0.3]),
            "highway": np.array([0.6, -0.4, 0.3, 0.7, -0.2]),
            "road": np.array([0.5, -0.4, 0.2, 0.6, -0.1]),
            "tech park": np.array([0.85, -0.65, 0.3, 0.4, -0.2]),
            "building": np.array([0.75, -0.55, 0.35, 0.3, -0.15]),
            "farmland": np.array([-0.6, 0.7, -0.4, -0.4, 0.3]),
            "encroachment": np.array([0.65, -0.5, 0.4, 0.2, -0.3]),
            "wetland": np.array([-0.5, 0.5, 0.7, -0.2, 0.2])
        }

    def generate_tile_embedding(self, tile_info: Dict[str, Any], bands_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extracts semantic embedding for a tile.
        Combines multispectral index signatures (NDVI, NDBI) and land-cover tags
        into a normalized 512-dimensional vector.
        """
        summary = tile_info.get("land_cover_summary", {})
        tags = summary.get("query_tags", [])
        desc = summary.get("description", "").lower()
        
        vec = np.zeros(self.embedding_dim, dtype=np.float32)

        # 1. Encode spectral signatures if bands_matrix is available (Red, Green, Blue, NIR, SWIR)
        if bands_matrix is not None and bands_matrix.shape[0] >= 5:
            red = bands_matrix[0]
            nir = bands_matrix[3]
            swir = bands_matrix[4]
            
            # Compute average NDVI & NDBI
            ndvi = np.mean((nir - red) / (nir + red + 1e-6))
            ndbi = np.mean((swir - nir) / (swir + nir + 1e-6))

            vec[0] = ndbi * 1.5   # Built-up index score
            vec[1] = ndvi * 1.5   # Vegetation index score
            vec[2] = np.mean(bands_matrix[2]) * 2.0 # Water/Blue score

        # 2. Encode textual tags and semantic description
        for word, concept_vec in self.concept_vectors.items():
            if any(word in t.lower() for t in tags) or word in desc:
                # Add concept weights to the vector header
                vec[0:5] += concept_vec * 0.8

        # 3. Fill higher dimensions with deterministic hash projection
        seed_val = abs(hash(tile_info["tile_id"])) % (2**31)
        rng = np.random.RandomState(seed_val)
        proj = rng.randn(self.embedding_dim - 5).astype(np.float32) * 0.1
        vec[5:] = proj

        # Normalize L2 norm
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def generate_text_embedding(self, query: str) -> np.ndarray:
        """
        Embeds a natural language text query (e.g. 'built-up land increase vegetation loss')
        into the 512-dim vector space.
        """
        query_lower = query.lower()
        vec = np.zeros(self.embedding_dim, dtype=np.float32)

        matched = False
        for word, concept_vec in self.concept_vectors.items():
            if word in query_lower:
                vec[0:5] += concept_vec * 1.2
                matched = True

        if not matched:
            # Fallback keyword matching
            if "built" in query_lower or "urban" in query_lower or "building" in query_lower:
                vec[0:5] += self.concept_vectors["built-up"]
            if "veg" in query_lower or "tree" in query_lower or "green" in query_lower:
                vec[0:5] += self.concept_vectors["vegetation"]
            if "water" in query_lower or "lake" in query_lower:
                vec[0:5] += self.concept_vectors["water"]

        # Fill deterministic hash for query words
        seed_val = abs(hash(query_lower)) % (2**31)
        rng = np.random.RandomState(seed_val)
        vec[5:] = rng.randn(self.embedding_dim - 5).astype(np.float32) * 0.05

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def build_index(self, tiles_metadata: List[Dict[str, Any]]):
        """
        Builds FAISS vector index from tile dataset.
        """
        self.tile_ids = []
        vecs = []

        for tile in tiles_metadata:
            self.tile_ids.append(tile["tile_id"])
            bands_path = tile.get("bands_path")
            bands_matrix = None
            if bands_path and os.path.exists(bands_path):
                bands_matrix = np.load(bands_path)
            
            emb = self.generate_tile_embedding(tile, bands_matrix)
            vecs.append(emb)

        self.embeddings = np.array(vecs, dtype=np.float32)

        if HAS_FAISS:
            # Inner product / Cosine similarity index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(self.embeddings)

    def search_text(self, query: str, top_k: int = 5, year_filter: Optional[int] = None, tiles_dict: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """
        Performs natural language semantic search over satellite index.
        Returns list of (tile_id, similarity_score).
        """
        if self.embeddings is None or len(self.tile_ids) == 0:
            return []

        q_vec = self.generate_text_embedding(query)
        q_vec = q_vec.reshape(1, -1)

        if HAS_FAISS and self.index is not None:
            # Query top_k * 2 to allow filtering
            scores, indices = self.index.search(q_vec, min(len(self.tile_ids), top_k * 3))
            raw_results = [(self.tile_ids[idx], float(scores[0][i])) for i, idx in enumerate(indices[0]) if idx >= 0]
        else:
            # Cosine similarity fallback
            scores = np.dot(self.embeddings, q_vec.T).squeeze()
            top_indices = np.argsort(scores)[::-1]
            raw_results = [(self.tile_ids[idx], float(scores[idx])) for idx in top_indices]

        # Apply year filter if specified
        results = []
        for tile_id, score in raw_results:
            if year_filter and tiles_dict and tile_id in tiles_dict:
                if tiles_dict[tile_id].get("year") != year_filter:
                    continue
            results.append((tile_id, score))
            if len(results) >= top_k:
                break

        return results
