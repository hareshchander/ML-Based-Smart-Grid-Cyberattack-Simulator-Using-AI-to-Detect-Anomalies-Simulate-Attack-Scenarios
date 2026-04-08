from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import numpy as np

def train_isolation_forest(X_train, contamination=0.1):
    """
    Trains an Isolation Forest model for unsupervised anomaly detection.
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_train)
    return model

def evaluate_isolation_forest(model, X_test, y_true):
    """
    Evaluates Isolation Forest model. 
    Converts model outputs (1: normal, -1: anomaly) to (0: normal, 1: anomaly).
    """
    y_pred = model.predict(X_test)
    y_pred = [1 if x == -1 else 0 for x in y_pred]
    
    metrics = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': metrics[0],
        'recall': metrics[1],
        'f1': metrics[2],
        'predictions': y_pred
    }

def train_mlp_model(X_train, y_train):
    """
    Trains a Multi-layer Perceptron (MLP) as a robust, supervised alternative to LSTM.
    """
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_mlp_model(model, X_test, y_true):
    """
    Evaluates MLP model performance.
    """
    y_pred = model.predict(X_test)
    
    metrics = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': metrics[0],
        'recall': metrics[1],
        'f1': metrics[2],
        'predictions': y_pred
    }
