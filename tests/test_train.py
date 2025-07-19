import json
import pytest
from sklearn.linear_model import LogisticRegression
from src.train import config, model

def test_config_loading():
    # Check required keys
    assert "C" in config
    assert "solver" in config
    assert "max_iter" in config
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

def test_model_type():
    # Check model type
    assert isinstance(model, LogisticRegression)

def test_model_is_trained():
    # Check model is fitted
    assert hasattr(model, "coef_")
    assert hasattr(model, "classes_")
