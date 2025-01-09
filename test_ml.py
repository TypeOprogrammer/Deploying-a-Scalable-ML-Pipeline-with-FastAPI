import pytest
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from ml.model import train_model
import numpy as np
from ml.model import inference  
from sklearn.ensemble import RandomForestClassifier



# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_inference_return_type(trained_model, example_data):
    """
    Test that inference returns a numpy array (which is the expected type).
    """
    X_train, _ = example_data
    preds = inference(trained_model, X_train)
    

    
    
    assert isinstance(preds, np.ndarray), "Inference should return a numpy array."
    assert preds.ndim == 1, "The predictions array should be 1-dimensional."


# TODO: implement the second test. Change the function name and input as needed

    """
    # Compute metrics for model performance
    """
def compute_model_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    return accuracy, precision, recall, f1


# TODO: implement the third test. Change the function name and input as needed

def is_it_rf():
    """
    Check if trained model uses random forest
    """
    
    X_train = np.random.rand(75, 5)
    y_train = np.random.randint(0, 2, 75)
    
    model = train_model(X_train, y_train)
    
    #returns false if model is not random forest
    assert isinstance(model, RandomForestClassifier), "False"






