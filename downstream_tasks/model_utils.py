from batch_processing import embed_for_linear_probing
from evaluation import logistic_regression_analysis
from coati.models.io import load_e3gnn_smiles_clip_e2e, load_coati2
from typing import List, Dict, Any

import torch
from torch import nn
import clamp
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional


def load_models(device: torch.device):
    """
    Loads various machine learning models and returns them in a dictionary for easy access by name.

    Args:
    device (torch.device): The device (CPU or GPU) to load the models onto.

    Returns:
    dict: A dictionary containing all models and tokenizers, accessible by their names.
    """

    models = {}

    """ 
    Grande Close Model:
        - Loss = InfoNCE + AR (AR is the autoregressive entropy loss)
        - E(3)-GNN = 5*256
        - Transformer = 16\*16\*256
        - Latent Dim. = 256
        - url = s3://terray-public/models/grande_closed.pkl 
    """

    models['coati_grande_encoder'], models['coati_grande_tokenizer'] = load_e3gnn_smiles_clip_e2e(
        freeze=True, 
        device=device, 
        doc_url="s3://terray-public/models/grande_closed.pkl"
    )

    """ 
    Autoregressive only Model:
        - Loss = AR
        - E(3)-GNN = N/A
        - Transformer = 16\*16\*256
        - Latent Dim. = 256
        - url = s3://terray-public/models/autoreg_only.pkl 
    """

    models['coati_autoreg_encoder'], models['coati_autoreg_tokenizer'] = load_e3gnn_smiles_clip_e2e(
        freeze=True, 
        device=device, 
        doc_url="s3://terray-public/models/autoreg_only.pkl"
    )

    """ 
    COATI2 Model:
        - trained on ~2x more data
        - Loss = InfoNCE + AR
        - chiral-aware 3D GNN = 5*256? (code not available)
        - Transformer = 16\*16\*256
        - Latent Dim. = 512 (new!)
        - url = s3://terray-public/models/coati2_chiral_03-08-24.pkl 
        """

    models['coati2_encoder'], models['coati2_tokenizer'] = load_coati2(
        freeze=True, 
        device=device, 
        doc_url="s3://terray-public/models/coati2_chiral_03-08-24.pkl"
    )

    """
    CLAMP Model:
        - Compound Encoder = Input:8192x4096, Hidden:4096x2048, Output:2048x768
        - Assay Encoder = Input:512x4096, Hidden:4096x2048, Output:2048x768
        - Latent Dim. = 768
    """

    models['clamp_model ']= clamp.CLAMP(device='cpu')
    models['clamp_model'].eval()

    return models

def execute_linear_probing(data_records: List[Dict[str, Any]], combined_df: pd.DataFrame, model_details: Dict[str, Any], dataset_name: str) -> None:
    """
    Processes a given model: Computes embeddings, performs logistic regression analysis, and saves results.

    Args:
        data_records (List[Dict[str, Any]]): A list of dictionaries, each containing data record information.
        combined_df (pd.DataFrame): The combined DataFrame containing all necessary data for processing.
        model_details (Dict[str, Any]): Dictionary containing model-related data like the model itself, tokenizer (if applicable), and model name.
        dataset_name (str): The name of the dataset currently being processed, used for file naming.

    Returns:
        None: This function performs operations in-place and saves files to disk without returning any value.
    """
    model_name, encoder, tokenizer = model_details['name'], model_details['encoder'], model_details.get('tokenizer')
    embeddings, labels, indices = embed_for_linear_probing(
        data_records,
        model_name=model_name,
        encoder=encoder,
        tokenizer=tokenizer
    )

    # Perform logistic regression for default and multiple scaffold splits
    for i in range(11):  # Includes default (0-9) and multi-scaffold (not specified as 10 but logical extension)
        split_key = 'scaffold_split' if i == 0 else f'scaffold_split_{i-1}'
        train_idx = np.where(combined_df[split_key] == 'train')[0]
        test_idx = np.where(combined_df[split_key] == 'test')[0]
        logistic_regression_analysis(embeddings, labels, train_idx, test_idx, dataset_name=f'{dataset_name}_{split_key}', model_name=model_name)

    # Save embeddings, indices, and labels
    np.save(f'{encoder}_{dataset_name}_embeddings.npy', embeddings)
    np.save(f'{encoder}_{dataset_name}_indices.npy', indices)
    labels_df = pd.DataFrame(labels, columns=['Label'])
    labels_df.to_csv(f'{encoder}_{dataset_name}_labels.csv', index=False)