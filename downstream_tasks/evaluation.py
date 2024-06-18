import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from pandas as pd
from typing import Callable, Dict, Any
from coati.models.regression.basic_due import basic_due

def bootstrap_metric(function, n: int=500):
    "wrapper for metrics to bootstrap e.g. calc std"
    def wrapper(y_true, y_pred, sample_weight=None):
        l = (len(y_true))
        res = []
        for i in range(n):
            s = np.random.choice(range(l), l, replace=True)
            if not len(np.unique(y_true[s]))==2:
                continue
            else:
                res.append( function(y_true[s], y_pred[s], sample_weight=None if sample_weight is None else sample_weight[s]))#,
        return np.array(res)
    return wrapper

def davgp_score(y_true, y_pred, sample_weight=None):
    avgp = average_precision_score(y_true, y_pred, sample_weight=sample_weight)
    y_avg = np.average(y_true, weights=sample_weight)
    return avgp - y_avg

def perform_model_analysis(embeddings: np.ndarray, labels: np.ndarray, indices: np.ndarray,
                           train_idx: np.ndarray, test_idx: np.ndarray,
                           analysis_type: str, dataset_name: str,
                           model_name: str, model_details: Dict[str, Any]= None, 
                           results_file: str = 'analysis_results.csv'):
    """
    Perform model analysis using either logistic regression or DUE model.

    Args:
        embeddings (np.ndarray): The embeddings of the data.
        labels (np.ndarray): The labels corresponding to the embeddings.
        indices (np.ndarray): Indices used for additional reference or splits.
        train_idx (np.ndarray): Indices for training data.
        test_idx (np.ndarray): Indices for testing data.
        model_details (Dict[str, Any]): Details and callable functions for model analysis.
        dataset_name (str): Name of the dataset being analyzed.
        model_name (str): Name of the model used for embedding.
        results_file (str): Path to save the CSV results.
        analysis_type (str): Type of analysis to perform ('logistic_regression' or 'due').
    """
    
    X_train = embeddings[train_idx]
    y_train = labels[train_idx]
    X_test = embeddings[test_idx]
    y_test = labels[test_idx]

    if model_details is None:
        if analysis_type == 'logistic_regression':
            model_details = {
                'max_iter': 1500,
                'class_weight': 'balanced',
                'C': 1,
                'random_state': 70135
            }
        elif analysis_type == 'due':
            model_details = {
                'x_field': "emb_smiles",
                'y_field': "qed",
                'continue_training': True,
                'steps': 1e4
            }
        else:
            raise ValueError("Unsupported analysis type")

    if analysis_type == 'logistic_regression':
        model = LogisticRegression(**model_details['params'])
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(embeddings[test_idx])[:, 1]
    elif analysis_type == 'due':
        model, model_results = basic_due(embeddings, **model_details['params'])
        model = model.to('cpu')
        y_pred, y_true, uncertainties = model_results
    else:
        raise ValueError("Unsupported analysis type")

    # Calculating scores
    AUROC_score = roc_auc_score(y_test, y_pred)
    roc_auc_std = bootstrap_metric(roc_auc_score, n=500)(y_test, y_pred).std()

    avgp = average_precision_score(y_test, y_pred)
    avgp_std = bootstrap_metric(average_precision_score, n=500)(y_test, y_pred).std()

    davgp = avgp - y_test.mean()
    davgp_std = bootstrap_metric(davgp_score, n=500)(y_test, y_pred).std()

    # Print the results to console
    print(f"dAP={davgp:.3f}, AUROC={AUROC_score:.3f}")

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Dataset': [dataset_name],
        'Analysis Type': [analysis_type],
        'Model': [model_name],
        'AUROC Score': [AUROC_score],
        'AUROC std': [roc_auc_std],
        'avgp': [avgp],
        'avgp_std': [avgp_std],
        'davgp': [davgp],
        'davgp_std': [davgp_std]
    })

    # Save or append the results to a CSV file
    # Check if the file exists to decide whether to write headers or not
    import os
    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_file, mode='w', header=True, index=False)