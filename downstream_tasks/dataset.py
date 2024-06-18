import os
import subprocess
from typing import List, Tuple

def download_and_extract_data(url: str, target_dir: str) -> None:
    """
    Download and extract dataset from a given URL.
    
    Parameters:
    url (str): URL to the dataset zip file.
    target_dir (str): Directory path where the dataset will be stored.
    """
    os.makedirs(target_dir, exist_ok=True)

    # Check if the dataset appears to be already downloaded
    if os.listdir(target_dir):
        print("Dataset already downloaded and extracted.")
        return
    
    subprocess.run(f"wget -N -r {url} -O {os.path.join(target_dir, 'downstream.zip')}", shell=True)
    subprocess.run(f"unzip {os.path.join(target_dir, 'downstream.zip')} -d {target_dir}; rm {os.path.join(target_dir, 'downstream.zip')}", shell=True)
    
    print("Download and extraction complete.")

def list_datasets(base_path: str) -> List[Tuple[str, str, str]]:
    """
    List available datasets and their corresponding paths for activity and compound SMILES files.

    Parameters:
    base_path (str): Base directory containing the extracted datasets.

    Returns:
    List[Tuple[str, str, str]]: A list of tuples, each containing the name of the dataset and the paths to activity and compound SMILES files.
    """
    dataset_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    datasets = []
    for dataset in dataset_dirs:
        activity_path = os.path.join(base_path, dataset, 'activity.parquet')
        smiles_path = os.path.join(base_path, dataset, 'compound_smiles.parquet')
        if os.path.exists(activity_path) and os.path.exists(smiles_path):
            datasets.append((dataset, activity_path, smiles_path))
    return datasets
