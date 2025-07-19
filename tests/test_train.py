import json
import joblib
import os
import pytest
from sklearn.linear_model import LogisticRegression

# Load config
with open("config/config.json") as f:
    config = json.load(f)

# Load model
model_path = "model_train.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

def test_config_keys():
    assert "C" in config
    assert "solver" in config
    assert "max_iter" in config

def test_model_type():
    assert isinstance(model, LogisticRegression)

def test_model_attributes():
    assert hasattr(model, "predict")
    assert hasattr(model, "coef_")
