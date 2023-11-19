import numpy as np


def compute_semantic_hard_boundary(
        t_id: np.ndarray,
        num_classes: int,
        bd_width: int = 30) -> np.ndarray:
    """
    Args:
        t_id (np.ndarray): 1d array of activity class IDs. shape=(T,)
        num_classes (int): -
        bd_width (int): boundary width
    Returns:
        t (np.ndarray): shape = (N_CLASSES,T)
    """
    assert t_id.ndim == 1
    seq_len = len(t_id)

    change_point = np.where(np.abs(t_id[1:] - t_id[:-1]) != 0)[0]
    bd = np.zeros((num_classes * 2, seq_len)).astype(np.int64)

    for ind in change_point:
        # boundary (tail)
        cls_prev = t_id[ind]
        head = max(ind - bd_width // 2, 0)
        tail = min(ind + bd_width // 2, seq_len - 1) + 1
        bd[cls_prev * 2 + 1, head:tail] = 1

        # boundary (head)
        cls_next = t_id[ind + 1]
        head = max(ind + 1 - bd_width // 2, 0)
        tail = min(ind + 1 + bd_width // 2, seq_len - 1) + 1
        bd[cls_next * 2, head:tail] = 1

    return bd