from dataset import download_and_extract_data, list_datasets
from data_utils import prepare_data
from model_utils import load_models, execute_linear_probing

import numpy as np
import os
import torch
import pandas as pd

def main():
    base_url = "https://cloud.ml.jku.at/s/pyJMm4yQeWFM2gG/download"
    data_directory = './data'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Download and prepare datasets
    download_and_extract_data(base_url, data_directory)
    datasets = list_datasets(os.path.join(data_directory, 'downstream'))

    # Load models
    models = load_models(device)

    # Dictionary to map model configurations
    model_configurations = {
        'coati_grande': {'encoder': models['coati_grande_encoder'], 'tokenizer': models['coati_grande_tokenizer']},
        'coati_autoreg': {'encoder': models['coati_autoreg_encoder'], 'tokenizer': models['coati_autoreg_tokenizer']},
        'clamp': {'encoder': models['clamp_model']}  # No tokenizer needed
    }

    # Process each dataset with each model configuration
    for dataset_name, activity_path, smiles_path in datasets:
        combined_df = prepare_data(smiles_path, activity_path)
        data_records = combined_df.to_dict('records')

        for model_key, model_details in model_configurations.items():
            print(f"Processing {model_key} for dataset {dataset_name}")
            model_details['name'] = model_key  # Include model name in details for clarity in processing
            execute_linear_probing(data_records, combined_df, model_details, dataset_name)

if __name__ == "__main__":
    main()
