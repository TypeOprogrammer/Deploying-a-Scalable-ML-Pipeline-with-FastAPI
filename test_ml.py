import pytest
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Test 1: Check if process_data returns the expected type for X and y
def test_process_data():
    """
    Test function returns numpy arrays for X and y.
    """
    # Dummy data 
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    X_processed, y_processed, _, _ = process_data(X, categorical_features=[], label=None, training=True)
    assert isinstance(X_processed, np.ndarray), "X should be a numpy array"
    assert isinstance(y_processed, np.ndarray), "y should be a numpy array"

# Test 2: Check if the model is a RandomForestClassifier
def test_train_model():
    """
    Test if the trained model is a RandomForestClassifier.
    """
    
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model is RandomForestClassifier"

# Test 3: Check if compute_model_metrics returns valid float values for precision, recall, and F1
def test_compute_model_metrics():
    """
    Test if compute_model_metrics returns precision, recall, and F1 as floats.
    """
    
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert isinstance(precision, float), "Float"
    assert isinstance(recall, float), "Float"
    assert isinstance(fbeta, float), "Float"
