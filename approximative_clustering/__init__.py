from typing import Callable, TypeVar, Optional

import networkx as nx
import numpy as np


_X = TypeVar("_X")


class ApproximativeClustering:

    def __init__(self, similarity: Callable[[_X, _X], float], k: int = 20_000, eps: float = 0.5, rng: np.random.Generator = np.random.default_rng()):
        self.similarity = similarity
        self.k = k
        self.eps = eps
        self.rng = rng

    def cluster_louvain(self, X: np.ndarray) -> np.ndarray:
        graph = nx.Graph()
        for i in range(X.shape[0]):
            graph.add_node(i)
        for i in range(X.shape[0]):
            for j in range(i, X.shape[0]):
                graph.add_edge(i, j, weight=self.similarity(X[i, :], X[j, :]))
        labels = np.zeros(X.shape[0], dtype=int)
        for i, community in enumerate(nx.community.louvain_communities(graph, threshold=self.eps, seed=1)):
            for c in community:
                labels[c] = i
        return labels

    def fit_predict(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        self.labels_ = -np.ones(X.shape[0], dtype=int)

        usable = np.ones(X.shape[0], dtype=bool)
        while np.any(usable):
            part_idx = self.sample_idx(usable)
            part_X = X[part_idx, :]

            cluster_ids = self.cluster_louvain(part_X) + np.max(self.labels_[~usable], initial=0)
            self.labels_[part_idx] = cluster_ids
            usable[part_idx] = False

            for i, s in zip(np.where(usable)[0], X[usable, :]):
                c = self.find_nn_cluster_id(s, part_X, cluster_ids)
                if c is not None:
                    self.labels_[i] = c
                    usable[i] = False

        return self.labels_

    def find_nn_cluster_id(self, s, cluster_data: np.ndarray, cluster_ids: np.ndarray) -> Optional[int]:
        clusters, cluster_counts = np.unique(cluster_ids, return_counts=True)
        similarities = np.zeros(clusters.shape[0], dtype=float)
        for i, x in zip(cluster_ids, cluster_data):
            similarities[clusters == i] += self.similarity(s, x)
        similarities = similarities / cluster_counts
        max_sim = np.nanmax(similarities, initial=np.nan)
        if np.isnan(max_sim) or max_sim < self.eps:
            return None
        return clusters[np.nanargmax(similarities)]

    def sample_idx(self, usable: np.ndarray) -> np.ndarray:
        idx = np.arange(usable.shape[0])[usable]
        self.rng.shuffle(idx)
        return idx[:self.k]