"""Cluster a seed's outputs by semantic / structural / behavioural similarity.

Stage 4 of the pipeline: two outputs join the same cluster when their similarity
meets ``cluster_beta``.  Transitive grouping is done by single-linkage
(connected components via union-find) over the thresholded similarity graph, so a
chain of pairwise-equivalent outputs forms one cluster.

The similarity is the **output-to-output** equivalence used by the spec's
``sim(G_i, G_j)`` — equivalence of the *outputs themselves*, independent of which
prompt produced them — so a neutral (empty) query is passed to the backend.
"""

from __future__ import annotations

from typing import List, Sequence

from casa.similarity.base import SimilarityBackend

_NEUTRAL_QUERY = ""


class _UnionFind:
    """Minimal disjoint-set structure."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def similarity_matrix(
    backend: SimilarityBackend, outputs: Sequence[str]
) -> List[List[float]]:
    """Symmetric ``N x N`` output-to-output similarity matrix (diagonal 1.0).

    Args:
        backend: The similarity backend.
        outputs: The ``N`` generated strings.

    Returns:
        A list-of-lists similarity matrix.
    """
    n = len(outputs)
    mat = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = backend.similarity(_NEUTRAL_QUERY, outputs[i], outputs[j])
            mat[i][j] = mat[j][i] = s
    return mat


def cluster_outputs(matrix: List[List[float]], beta: float) -> List[List[int]]:
    """Single-linkage clusters of output indices at threshold ``beta``.

    Args:
        matrix: Symmetric similarity matrix from :func:`similarity_matrix`.
        beta: Similarity threshold for joining two outputs.

    Returns:
        A list of clusters (each a list of output indices), ordered by
        descending size then ascending first index for determinism.
    """
    n = len(matrix)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] >= beta:
                uf.union(i, j)
    groups: dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    clusters = list(groups.values())
    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters
