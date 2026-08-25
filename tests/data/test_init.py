# tests/data/test_init.py
import importlib


def test_module_exports():
    module = importlib.import_module("kitaev.data")
    assert "SupervisedKitaevDataset" in module.__all__
    assert "UnsupervisedMuGenerator" in module.__all__
