import torch
from torch import nn
from coati.common.util import batch_indexable
import numpy as np
import rdkit
from rdkit import Chem
from typing import List, Tuple, Dict, Any
from utilities import save_checkpoint, find_latest_checkpoint, load_checkpoint

def embed_for_linear_probing(
        records: List[Dict], 
        model_name: str, 
        encoder: nn.Module, 
        tokenizer: nn.Module=None, 
        batch_size: int = 128, 
        smiles_field: str = "smiles"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process records in batches to compute embeddings and optionally RDKit properties."""

    embeddings, labels, indices = [], [], []
    batch_iter = batch_indexable(records, batch_size)

    with torch.no_grad():
        for i, batch in enumerate(batch_iter):
            print(f"Processing batch {i+1}/{len(records)//batch_size}")

            try:
                batch_mols = [Chem.MolFromSmiles(row[smiles_field]) for row in batch]
                batch_smiles = [Chem.MolToSmiles(m) for m in batch_mols]
                if model_name == 'coati':
                    batch_tokens = torch.tensor(
                        [tokenizer.tokenize_text("[SMILES]" + s + "[STOP]", pad=True) if s != "*" else tokenizer.tokenize_text("[SMILES]C[STOP]", pad=True) for s in batch_smiles],
                        device=encoder.device,
                        dtype=torch.int,
                    )
                    batch_embeds = encoder.encode_tokens(batch_tokens, tokenizer).detach().cpu().numpy()
                elif model_name == 'clamp':
                    batch_tokens = [s if s != '*' else 'C' for s in batch_smiles]
                    batch_embeds = encoder.encode_smiles(batch_tokens).detach().cpu().numpy()

                embeddings.append(batch_embeds)
                labels.append([row['activity'] for row in batch])
                indices.append([row['compound_idx'] for row in batch])

            except Exception as Ex:
                print(Ex)
                continue

    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    indices = np.concatenate(indices)

    return embeddings, labels, indices

def embed_for_due(
        records: List[Dict[str, Any]],
        model_name: str ,
        encoder: nn.Module,
        tokenizer: Any = None,
        batch_size: int = 128,
        score: bool = True,
        smiles_field: str = "smiles",
        checkpoint_base: str = "checkpoint",
        checkpoint_interval: int = 1000
    ) -> List[Dict[str, Any]]:
    """
    Computes embeddings and optional RDKit properties for molecules in batches, resuming from the last checkpoint if available.

    Args:
        records (List[Dict[str, Any]]): A list of dictionary records where each record contains molecule data.
        model_name (str): The name of the model to use for embedding generation.
        encoder (nn.Module): The PyTorch model that performs the embedding.
        tokenizer (Any): The tokenizer to process smile strings into a format suitable for the encoder.
        batch_size (int): The number of records to process in each batch.
        score (bool): If True, calculate RDKit properties such as LogP and QED for each molecule.
        smiles_field (str): The key in the records dict that contains the SMILES string.
        checkpoint_base (str): Base name for checkpoint files.
        checkpoint_interval (int): The number of batches between saving checkpoints.

    Returns:
        List[Dict[str, Any]]: The updated list of records with embeddings and, if requested, RDKit properties added.
    """

    # Determine the latest checkpoint
    latest_checkpoint, starting_batch_index = find_latest_checkpoint(checkpoint_base)
    if latest_checkpoint:
        records = load_checkpoint(latest_checkpoint)
        print(f"Resuming from checkpoint {starting_batch_index}, batch {starting_batch_index * batch_size}")

    num_batches = len(records) // batch_size
    batch_iter = batch_indexable(records, batch_size)

    with torch.no_grad():
        for i, batch in enumerate(batch_iter, start=starting_batch_index):
            print(f"batch: {i}/{num_batches}")
            if model_name == 'coati':
              try:
                  batch_mols = [Chem.MolFromSmiles(row[smiles_field]) for row in batch]
                  batch_smiles = [Chem.MolToSmiles(m) for m in batch_mols]
                  batch_tokens = torch.tensor(
                      [
                          tokenizer.tokenize_text("[SMILES]" + s + "[STOP]", pad=True)
                          if s != "*"
                          else tokenizer.tokenize_text("[SMILES]C[STOP]", pad=True)
                          for s in batch_smiles
                      ],
                      device=encoder.device,
                      dtype=torch.int,
                  )
                  #print(batch_tokens[0])
                  batch_embeds = encoder.encode_tokens(batch_tokens, tokenizer)
                  #print(batch_embeds[0]) #1x256
                  if score:
                      batch_logp = [rdkit.Chem.Crippen.MolLogP(m) for m in batch_mols]
                      batch_qed = [rdkit.Chem.QED.qed(m) for m in batch_mols]
                  if len(batch) < 2:
                      batch[0]["emb_smiles"] = batch_embeds[0].detach().cpu().numpy()
                      if score:
                          batch[0]["qed"] = batch_qed[0]
                          batch[0]["logp"] = batch_logp[0]
                          batch[0]["smiles"] = batch_smiles[0]
                  else:
                      for k, r in enumerate(batch):
                          batch[k]["emb_smiles"] = batch_embeds[k].detach().cpu().numpy()
                          if score:
                              batch[k]["qed"] = batch_qed[k]
                              batch[k]["logp"] = batch_logp[k]
                              batch[k]["smiles"] = batch_smiles[k]
              except Exception as Ex:
                  print(Ex)
                  continue

            if model_name == 'clamp':
              try:
                  batch_mols = [Chem.MolFromSmiles(row[smiles_field]) for row in batch]
                  batch_smiles = [Chem.MolToSmiles(m) for m in batch_mols]
                  batch_tokens = [s if s != '*' else 'C' for s in batch_smiles]
                  batch_embeds = encoder.encode_smiles(batch_tokens)
                  #print(batch_embeds[0]) #1x768
                  if score:
                      batch_logp = [rdkit.Chem.Crippen.MolLogP(m) for m in batch_mols]
                      batch_qed = [rdkit.Chem.QED.qed(m) for m in batch_mols]
                  if len(batch) < 2:
                      batch[0]["emb_smiles"] = batch_embeds[0].detach().cpu().numpy()
                      if score:
                          batch[0]["qed"] = batch_qed[0]
                          batch[0]["logp"] = batch_logp[0]
                          batch[0]["smiles"] = batch_smiles[0]
                  else:
                      for k, r in enumerate(batch):
                          batch[k]["emb_smiles"] = batch_embeds[k].detach().cpu().numpy()
                          if score:
                              batch[k]["qed"] = batch_qed[k]
                              batch[k]["logp"] = batch_logp[k]
                              batch[k]["smiles"] = batch_smiles[k]
              except Exception as Ex:
                  print(Ex)
                  continue
            # Save a checkpoint every 'checkpoint_interval' batches
            if (i + 1) % checkpoint_interval == 0:
                save_checkpoint(records, checkpoint_base, i + 1)
                print("checkpoint safed")

        # Final save to ensure the last part of data is also saved
        save_checkpoint(records, checkpoint_base, i + 1)

    return records

