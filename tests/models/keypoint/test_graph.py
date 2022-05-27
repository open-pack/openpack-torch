from openpack_torch.models.keypoint.graph import (NUM_NODES_MSCOCO,
                                                  NUM_NODES_NTU_RGBD,
                                                  get_adjacency_matrix)


def test_get_adjacency_matrix__01():
    hop_size = 2

    A = get_adjacency_matrix("MSCOCO", hop_size)

    assert A.shape[0] == hop_size + 1
    assert A.shape[1] == NUM_NODES_MSCOCO
    assert A.shape[2] == NUM_NODES_MSCOCO


def test_get_adjacency_matrix__02():
    hop_size = 2

    A = get_adjacency_matrix("NTU-RGBD", hop_size)

    assert A.shape[0] == hop_size + 1
    assert A.shape[1] == NUM_NODES_NTU_RGBD
    assert A.shape[2] == NUM_NODES_NTU_RGBD
