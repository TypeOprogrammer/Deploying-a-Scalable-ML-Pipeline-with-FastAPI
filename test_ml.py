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
    assert len(y_preds) > 0, "empty"
    assert all(isinstance(pred, (int, float)) for pred in y_preds), "All predictions should be numerical values"


