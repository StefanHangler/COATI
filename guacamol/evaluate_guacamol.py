#"FCD": fcd_value,
#"FCD GuacaMol": exp(-0.2 * fcd_value)

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from fcd_torch.fcd_torch.fcd import FCD
from rdkit.Chem import rdmolfiles, rdmolops

def get_device() -> torch.device:
    """Determine available device.

    Returns:
        device(type='cuda') if cuda is available, device(type='cpu') otherwise.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)

def calculate_validity(smiles_list):
    """ Check if SMILES are valid by attempting to create RDKit molecules. """
    valid = [Chem.MolFromSmiles(sm) is not None for sm in smiles_list]
    validity = np.mean(valid)
    return validity

def calculate_novelty(generated_smiles, training_smiles):
    """ Calculate novelty as the proportion of generated SMILES not in the training set. """
    training_set = set(training_smiles)
    novel = [sm not in training_set for sm in generated_smiles]
    novelty = np.mean(novel)
    return novelty

def calculate_uniqueness(smiles_list):
    """ Calculate uniqueness as the proportion of unique SMILES in the generated list. """
    unique_smiles = len(set(smiles_list))
    uniqueness = unique_smiles / len(smiles_list)
    return uniqueness

def calculate_fcd(generated_smiles: str, reference_smiles: str, canonicalize: bool = True, n_jobs: int = 8) -> float:
    """ Calculate the FCD between generated molecules and a reference set. """
    device = get_device()
    fcd_calculator = FCD(canonize=canonicalize, device=device, n_jobs=n_jobs, pbar=True)
    fcd_score = fcd_calculator(reference_smiles, generated_smiles)
    return fcd_score

def evaluate_guacamol(
    generated_smiles_path: str, 
    training_smiles_path: str = None, 
    reference_dataset_path: str = None, 
    model_name: str = 'COATI', 
    results_path: str = 'guacamol_results.csv'):
    """ Evaluate the generated SMILES against the training set or reference dataset. """
    
    with open(generated_smiles_path, "r") as f:
        generated_smiles = f.read().splitlines()
    
    if training_smiles_path is not None:
        with open(training_smiles_path, "r") as f:
            training_smiles = f.read().splitlines()

    # Calculate metrics
    validity = calculate_validity(generated_smiles)
    novelty = calculate_novelty(generated_smiles, training_smiles)
    uniqueness = calculate_uniqueness(generated_smiles)

    if reference_dataset_path is not None:
        with open(reference_dataset_path, "r") as f:
            reference_smiles = f.read().splitlines()
        fcd_score = calculate_fcd(generated_smiles, reference_smiles)
    elif training_smiles_path is not None:
        fcd_score = calculate_fcd(generated_smiles, training_smiles)
    else:
        raise ValueError("Either training_smiles_path or reference_dataset_path must be provided.")

    # Print results
    print("Validity:", validity)
    print("Novelty:", novelty)
    print("Uniqueness:", uniqueness)
    print("FCD Score:", fcd_score)
    print("FCD GuacaMol:", exp(-0.2 * fcd_score))

    # Store resutls into CSV (add if file exists)
    results = pd.DataFrame({
        "Model": model_name,
        "Validity": validity,
        "Novelty": novelty,
        "Uniqueness": uniqueness,
        "FCD Score": fcd_score,
        "FCD GuacaMol": exp(-0.2 * fcd_score)
    })

    if os.path.exists(results_path):
        results.to_csv(results_path, mode='a', header=False, index=False)
    else:
        results.to_csv(results_path, mode='w', header=True, index=False)

if __name__ == "__main__":



    evaluate_guacamol(
        generated_smiles_path="generated_smiles.smi",
        training_smiles_path="training_smiles.smi",
        reference_dataset_path="reference_dataset.smi",
        model_name="COATI",
        results_path="guacamol_results.csv"
    )