import pytest
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    save_model,
    load_model,
    performance_on_categorical_slice,
    train_model,
)



def test_inference():
    """
    Test inference of model
    """
    
    X = np.random.rand(20, 5)  
    y = np.random.randint(2, size=20)  

    
    model = train_model(X, y)  

    
    y_preds = inference(model, X)  

    
    assert y.shape == y_preds.shape, f"Expected shape to be {y.shape}, but got {y_preds.shape}"

    # 
    #assert len(y_preds) > 0, "empty"
    #assert all(isinstance(pred, (int, float)) for pred in y_preds), "All should be numerical values"


def test_compute_model_metrics():
    """
    Test compute_model_metrics
    """
    #to test range
    y_true = [1, 1, 0]
    y_preds = [0, 1, 1]

    
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)

    
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

    #  metrics should be within range [0, 1]
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
