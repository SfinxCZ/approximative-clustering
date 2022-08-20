from unittest.mock import Mock

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import given
from sklearn.datasets import make_blobs
from sklearn.metrics import homogeneity_score, completeness_score

from approximative_clustering import ApproximativeClustering


def euclidean(x, y) -> float:
    return 1 / (1 + np.sqrt((x - y) @ (x - y)))


@pytest.fixture(scope="module")
def clustering():
    return ApproximativeClustering(euclidean, k=50, eps=0.3)


@pytest.fixture(scope="function")
def blobs():
    return make_blobs(1000, centers=[[-3, -3], [3, 3], [-3, 3], [3, -3]], cluster_std=0.5)


@st.composite
def separable_data(draw, min_num_samples_per_cluster: int = 1, max_num_samples_per_cluster: int = 1000, min_num_clusters: int = 1, max_num_clusters: int = 10):
    n_features = draw(st.integers(min_value=1, max_value=5))
    n_samples_per_cluster = draw(
        st.lists(
            st.integers(min_value=min_num_samples_per_cluster, max_value=max_num_samples_per_cluster),
            min_size=min_num_clusters,
            max_size=max_num_clusters
        )
    )
    data = []
    labels = []
    for i, n_samples in enumerate(n_samples_per_cluster):
        data.append(draw(npst.arrays(dtype=np.float, shape=(n_samples, n_features),
                                     elements=st.floats(min_value=(i + 1) * 10, max_value=(i + 1) * 10 + 1, allow_nan=False, allow_infinity=False,
                                                        allow_subnormal=False))))
        labels.append(i * np.ones(n_samples, dtype=int))
    return np.vstack(data), np.hstack(labels)


def test_sample_idx():
    usable = np.ones(1000, dtype=bool)
    usable[1::2] = False
    generator = Mock(np.random.Generator)
    generator.shuffle = Mock(return_value=usable)
    clustering = ApproximativeClustering(euclidean, k=200, rng=generator)
    act_idx = clustering.sample_idx(usable)
    exp_idx = np.arange(0, 2 * clustering.k, 2)
    npt.assert_array_equal(act_idx, exp_idx)


@given(separable_data(), st.data())
def test_find_nn_cluster_id(clustering, s_data, draw):
    data, labels = s_data
    i = draw.draw(st.integers(min_value=0, max_value=data.shape[0] - 1))
    s = data[i, :] + 1e-2

    closest_cluster = clustering.find_nn_cluster_id(s, data, labels)
    assert closest_cluster == labels[i]


def test_cluster_louvain(clustering, blobs):
    data, labels = blobs
    act_labels = clustering.cluster_louvain(data)

    assert homogeneity_score(labels, act_labels) > 0.8
    assert completeness_score(labels, act_labels) > 0.8


def test_clustering(clustering, blobs):
    data, labels = blobs

    act_labels = clustering.fit_predict(data)
    assert homogeneity_score(labels, act_labels) > 0.8
    assert completeness_score(labels, act_labels) > 0.8
