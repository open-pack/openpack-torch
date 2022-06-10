from hydra.core.config_store import ConfigStore
from openpack_torch.configs._hydra import register_configs


def test_register_dataset_configs__01():
    register_configs()

    # checl
    cs = ConfigStore.instance()
    assert "user" in cs.repo.keys()
    print("user=", cs.repo["user"].keys())

    assert "dataset" in cs.repo.keys()
    assert "stream" in cs.repo["dataset"].keys()
    assert "split" in cs.repo["dataset"].keys()
    assert "annotation" in cs.repo["dataset"].keys()

    print("dataset/stream=", cs.repo["dataset"]["stream"].keys())
    print("dataset/split=", cs.repo["dataset"]["split"].keys())
    print("dataset/annotation=", cs.repo["dataset"]["annotation"].keys())
