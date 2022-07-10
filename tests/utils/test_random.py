from openpack_torch.utils import reset_seed


def test_reset_seed__01():
    seed = 0
    reset_seed(seed)
