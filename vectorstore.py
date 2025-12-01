# vectorstore.py 
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class VectorIndex:
    """
    Simple in-memory vector index using SentenceTransformers.
    This can be later replaced or extended with Pinecone integration.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents: List[str] = []
        self.embeddings: np.ndarray | None = None

    def build_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Build document representations from a DataFrame.
        Each column becomes a short textual description document.
        """
        docs: List[str] = []
        for col in df.columns:
            sample_values = (
                df[col]
                .dropna()
                .astype(str)
                .unique()[:5]  # up to 5 sample values
            )
            sample_str = ", ".join(sample_values) if len(sample_values) > 0 else "No non-null sample values"
            doc = (
                f"Column '{col}' (dtype: {df[col].dtype}). "
                f"Sample values: {sample_str}"
            )
            docs.append(doc)

        self.documents = docs
        if not docs:
            self.embeddings = None
            return

        self.embeddings = self.model.encode(
            docs,
            convert_to_numpy=True,
            show_progress_bar=False
        )

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Return top-k most similar documents to the query.
        """
        if not self.documents or self.embeddings is None:
            return []

        q_emb = self.model.encode([query], convert_to_numpy=True)[0]
        doc_embs = self.embeddings

        # cosine similarity
        dot = np.dot(doc_embs, q_emb)
        norms = np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(q_emb)
        sims = dot / (norms + 1e-10)

        idxs = np.argsort(sims)[::-1][:k]
        return [(self.documents[i], float(sims[i])) for i in idxs]


def build_vector_index_from_dataframe(df: pd.DataFrame) -> VectorIndex:
    """
    Helper to quickly create a VectorIndex from a DataFrame.
    """
    index = VectorIndex()
    index.build_from_dataframe(df)
    return index
