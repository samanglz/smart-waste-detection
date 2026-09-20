from typing import Tuple

import numpy as np


class EmbeddingSimilarity:
    """
    Utility for comparing image embeddings.

    This class is intentionally independent of:
        - YOLO
        - dataset splits
        - failure patterns
        - model evaluation
        - training

    Embeddings are expected to be 1-D numpy arrays.
    """

    @staticmethod
    def cosine_similarity(
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding_a:
                First embedding.

            embedding_b:
                Second embedding.

        Returns:
            Cosine similarity in the range [-1, 1].
        """

        a = np.asarray(
            embedding_a,
            dtype=np.float32,
        )

        b = np.asarray(
            embedding_b,
            dtype=np.float32,
        )

        if a.ndim != 1 or b.ndim != 1:
            raise ValueError(
                "Embeddings must be 1-D arrays."
            )

        if a.shape != b.shape:
            raise ValueError(
                "Embedding dimensions must match: "
                f"{a.shape} != {b.shape}"
            )

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0.0 or norm_b == 0.0:
            raise ValueError(
                "Cannot calculate cosine similarity "
                "for a zero-vector embedding."
            )

        return float(
            np.dot(a, b)
            / (norm_a * norm_b)
        )

    @staticmethod
    def rank(
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Rank candidate embeddings by cosine similarity.

        Args:
            query_embedding:
                1-D query embedding.

            candidate_embeddings:
                2-D array with shape:
                [num_candidates, embedding_dim].

        Returns:
            Indices sorted from most similar to least similar.
        """

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        candidates = np.asarray(
            candidate_embeddings,
            dtype=np.float32,
        )

        if query.ndim != 1:
            raise ValueError(
                "Query embedding must be 1-D."
            )

        if candidates.ndim != 2:
            raise ValueError(
                "Candidate embeddings must be 2-D."
            )

        if candidates.shape[1] != query.shape[0]:
            raise ValueError(
                "Embedding dimensions do not match: "
                f"{query.shape[0]} != "
                f"{candidates.shape[1]}"
            )

        query_norm = np.linalg.norm(query)

        if query_norm == 0.0:
            raise ValueError(
                "Query embedding cannot be a zero vector."
            )

        candidate_norms = np.linalg.norm(
            candidates,
            axis=1,
        )

        if np.any(candidate_norms == 0.0):
            raise ValueError(
                "Candidate embeddings contain "
                "a zero vector."
            )

        similarities = (
            candidates @ query
        ) / (
            candidate_norms * query_norm
        )

        return np.argsort(
            similarities
        )[::-1]

    @staticmethod
    def top_k(
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the top-k most similar candidates.

        Args:
            query_embedding:
                1-D query embedding.

            candidate_embeddings:
                2-D candidate embedding matrix.

            k:
                Number of nearest candidates.

        Returns:
            Tuple containing:

                indices:
                    Candidate indices sorted by similarity.

                similarities:
                    Corresponding cosine similarities.
        """

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        candidates = np.asarray(
            candidate_embeddings,
            dtype=np.float32,
        )

        if query.ndim != 1:
            raise ValueError(
                "Query embedding must be 1-D."
            )

        if candidates.ndim != 2:
            raise ValueError(
                "Candidate embeddings must be 2-D."
            )

        if candidates.shape[1] != query.shape[0]:
            raise ValueError(
                "Embedding dimensions do not match."
            )

        query_norm = np.linalg.norm(query)

        if query_norm == 0.0:
            raise ValueError(
                "Query embedding cannot be a zero vector."
            )

        candidate_norms = np.linalg.norm(
            candidates,
            axis=1,
        )

        if np.any(candidate_norms == 0.0):
            raise ValueError(
                "Candidate embeddings contain "
                "a zero vector."
            )

        similarities = (
            candidates @ query
        ) / (
            candidate_norms * query_norm
        )

        k = min(
            k,
            len(candidates),
        )

        indices = np.argsort(
            similarities
        )[::-1][:k]

        return (
            indices,
            similarities[indices],
        )

