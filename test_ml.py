import pytest
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Test 1: Check if process_data returns the expected type for X and y
def test_process_data():
    """
    Test if process_data function returns numpy arrays for X and y.
    """
    
    X = pd.DataFrame([[1, 2], [3, 4]], columns=['feature1', 'feature2'])
    y = np.array([0, 1])  

    X_processed, y_processed, _, _ = process_data(X, categorical_features=[], label=None, training=True)

    assert isinstance(X_processed, np.ndarray), "X should be a numpy array"
    assert isinstance(y_processed, np.ndarray), "y should be a numpy array"  

# Test 2: Check if RandomForestClassifier
def test_train_model():
    """
    Test if the trained model is a RandomForestClassifier.
    """
    
    X_train = pd.DataFrame([[1, 2], [3, 4]], columns=['feature1', 'feature2'])
    y_train = np.array([0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model is RandomForestClassifier"

# Test 3: Check if compute_model_metrics returns valid float values for precision, recall, and F1
def test_compute_model_metrics():
    """
    Test if compute_model_metrics returns precision, recall, and F1 as floats.
    """
    # Dummy data for testing
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float"
    assert isinstance(fbeta, float), "F1 should be a float"

