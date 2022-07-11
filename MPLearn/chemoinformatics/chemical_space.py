# -*- tab-width:4;indent-tabs-mode:nil;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=2 noet:

import os
import gzip
from typing import Sequence
import numpy as np
import pandas as pd
import tqdm
import rdkit.Chem
import rdkit.Chem.rdMolDescriptors
from rdkit.Chem import DataStructs

# for APDP pairs
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Sheridan
from rdkit import DataStructs

from .rdkit_support import Mol2MolSupplier


def download_huggingface_model(
    model_name,
    model_path,
    verbose = False):
    """
    Download and store Hugging Face model and tokenizer
    """

    from transformers import AutoTokenizer, AutoModelForCausalLM
    if verbose:
        print(f"Loading Hugging Face model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if not os.exists(model_path):
        if verbose:
            print(f"Output model path '{model_path}/{{model,tokenizer}}' does not exist, creating...")
        os.create(model_path)

    model.save_pretrained(
        f"{model_path}/model")
    tokenizer.save_pretrained(
        f"{model_path}/tokenizer")


def molecule_to_fingerprint_array(
    molecule,
    fingerprint_type,
    fingerprint_n_bits,
    verbose = False):
    """
    Conver rdkit molecule to numpy.array

    """

    if fingerprint_type == "ECFP4":
        fingerprint = np.zeros(fingerprint_n_bits, np.uint8)
        DataStructs.ConvertToNumpyArray(
            rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol=molecule, radius=2, nBits=fingerprint_n_bits),
            fingerprint)

    elif fingerprint_type == "APDP":
        ap_fp = Pairs.GetAtomPairFingerprint(molecule)
        dp_fp = Sheridan.GetBPFingerprint(molecule)
        #ap_fp.GetLength() == 8388608
        #dp_fp.GetLength() == 8388608
        #16777216 = 8388608 + 8388608

        fingerprint = np.zeros(fingerprint_n_bits)
        for i in ap_fp.GetNonzeroElements().keys():
            fingerprint[i % fingerprint_n_bits] = 1
        for i in dp_fp.GetNonzeroElements().keys():
            fingerprint[(i + 8388608) % fingerprint_n_bits] = 1
    else:
        raise ValueError(f"ERROR: Unrecognized fingerprint type '{fingerprint_type}'")
        exit(1)

    return fingerprint


def generate_fingerprints_smiles(
    smiles: Sequence[str],
    substance_ids: Sequence[str],
    fingerprint_type: str = 'ECFP4',
    fingerprint_n_bits: int = 1024,
    device: str = 'cpu',
    verbose: bool = False):
    """
    Generate fingerprints for a set of molecules

    Args:
        smiles: a list of smiles strings
        fingerprint_type: type of fingerprint to represent molecules for comparison
        fingerprint_n_bits: number of bits in the returned fingerprint
        device: specify hardware accelerator for models that support it
        model_path: for trained models, specify the model where to load the model
        verbose: verbose logging

    Returns:
        List[Dict[query_id:str, <library_fields>, tanimoto_similarity:float]]
    """
    valid_fingerprint_types = ('ECFP4', 'APDP', "huggingface")
    if not fingerprint_type.startswith(valid_fingerprint_types):
        raise ValueError((
            f"Unrecognized fingerprint_type '{fingerprint_type}'. ",
            f"Valid options are [{', '.join(valid_fingerprint_types)}]"))

    if isinstance(smiles, str):
        smiles = [smiles]

    if fingerprint_type.startswith("huggingface"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import selfies
        import torch
        model_path = fingerprint_type[12:]

        if verbose:
            print(f"Loading huggingface model '{model_path}' onto device '{device}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    fingerprints = []
    substance_ids_generated = []
    for index, substance_smiles in enumerate(tqdm.tqdm(smiles)):

        if fingerprint_type.startswith("huggingface"):
            try:
                substance_selfies = selfies.encoder(substance_smiles)
            except:
                print((
                    f"ERROR: Failed to generate selfies for molecule '{index}' ",
                    f"using fingerprint type '{fingerprint_type}' ",
                    f"with smiles '{substance_smiles}'; skipping..."))
                continue

            try:
                with torch.no_grad():
                    substance_tokens = tokenizer(substance_selfies)
                    input_ids = torch.tensor(substance_tokens.input_ids).to(device)
                    model_output = model.forward(input_ids = input_ids)
                    fingerprint = list(model_output.values())[0].mean(0)
                    fingerprint = fingerprint.cpu().detach().numpy()
            except:
                print((
                    f"ERROR: Failed to generate fingerprint for molecule '{index}' ",
                    f"using fingerprint type '{fingerprint_type}' ",
                    f"with smiles '{substance_smiles}'; skipping "))
                continue

        elif fingerprint_type in ["ECFP4", "APDP"]:
            try:
                molecule = rdkit.Chem.MolFromSmiles(substance_smiles, sanitize=False)
            except:
                print((
                    f"ERROR: RDKit failed to create molecule '{index}' ",
                    f"using fingerprint type '{fingerprint_type}' ",
                    f"with smiles '{substance_smiles}'; skipping..."))
                continue

            if molecule is None:
                print((
                    f"WARNING: RDKit failed to create molecule '{index}' ",
                    f"using fingerprint type '{fingerprint_type}' ",
                    f"with smiles '{substance_smiles}'; skipping..."))
                continue

            try:
                molecule.UpdatePropertyCache(strict=False)
                molecule = rdkit.Chem.AddHs(molecule, addCoords=True)
                molecule = rdkit.Chem.RemoveHs(molecule) # Also Sanitizes
            except ValueError as e:
                print((
                    f"ERROR: {str(e)}. Skipping molecule '{index}' ",
                    f"using fingerprint type '{fingerprint_type}' ",
                    f"with smiles '{smiles}'."))
                continue

            try:
                fingerprint = molecule_to_fingerprint_array(
                    molecule,
                    fingerprint_type,
                    fingerprint_n_bits,
                    verbose)
            except:
                print(f"ERROR: Unable to generate fingerprint for library molecule with index {index}")
                continue

        fingerprints.append(fingerprint)
        substance_ids_generated.append(substance_ids[index])

    fingerprints = np.array(fingerprints)
    return substance_ids_generated, fingerprints


# needs some debugging
def generate_fingerprints_sdf(
        library_path: str,
        fields: Sequence[str],
        fingerprint_type: str = 'ECFP4',
        fingerprint_n_bits: int = 1024,
        verbose=False):

    """
    Generate fingerprints for substances in the library

    Args:
        libray_path: path to a possibly gzipped library .sdf file
        fields: fields in the library .sdf file to be reported in the results
        fingerprint_type: type of fingerprint to represent molecules for comparison

    Returns:
        List[Dict[query_id:str, <library_fields>, tanimoto_similarity:float]]
    """

    # validate inputs
    if not os.path.exists(library_path):
        raise ValueError(f"The library path '{library_path}' does not exist.")

    valid_fingerprint_types = ['ECFP4', 'APDP']
    if fingerprint_type not in valid_fingerprint_types:
        raise ValueError((
            f"Unrecognized fingerprint_type '{fingerprint_type}'. ",
            f"Valid options are [{', '.join(valid_fingerprint_types)}]"))

    substances = []
    fingerprints = []
    if library_path.endswith(".gz"):
        supplier = rdkit.Chem.ForwardSDMolSupplier(gzip.open(library_path))
    else:
        supplier = rdkit.Chem.ForwardSDMolSupplier(library_path)

    for substance_index, substance in enumerate(supplier):
        try:
            fingerprint = molecule_to_fingerprint_array(
                substance,
                fingerprint_type,
                fingerprint_n_bits,
                verbose)
        except:
            print(f"WARNING: Unable to generate fingerprint for library molecule with index {substance_index}")
            continue
        fingerprints.append(fingerprint)

        substance_info = {}
        if fields is None:
            substance_info.update(substance.GetPropsAsDict())
        else:
            for field in fields:
                try:
                    field_value = substance.GetProp(field)
                    substance_info[field] = field_value
                except:
                    print(
                        f"WARNING: Library compound at index {substance_index} does not have field {field}.")
                    substance_info[field] = None
        substances.append(substance_info)

    if verbose:
        print(f"Found library contains {len(substances)} substances.")

    fingerprints = np.array(fingerprints)
    substances = pd.DataFrame(substances)

    return substances, fingerprints


# needs some debugging
def generate_fingerprints_mol2(
        library_path: str,
        fields: Sequence[str] = ["_Name"],
        fingerprint_type: str = 'ECFP4',
        fingerprint_n_bits: int = 1024,
        verbose=False):

    """
    Generate fingerprints for substances in the library

    Args:
        libray_path: path to library .mol2 file
        fields: fields in the library .mol2 file to be reported in the results
        fingerprint_type: type of fingerprint to represent molecules for comparison

    Returns:
        List[Dict[query_id:str, <library_fields>, tanimoto_similarity:float]]
    """

    # validate inputs
    if not os.path.exists(library_path):
        raise ValueError(f"The library path '{library_path}' does not exist.")

    valid_fingerprint_types = ["ECFP4", "APDP"]
    if fingerprint_type not in valid_fingerprint_types:
        raise ValueError((
            f"Unrecognized fingerprint_type '{fingerprint_type}'. ",
            f"Valid options are [{', '.join(valid_fingerprint_types)}]"))

    substances = []
    fingerprints = []
    for substance_index, (start_line, substance) in enumerate(Mol2MolSupplier(library_path)):
        try:
            fingerprint = molecule_to_fingerprint_array(
                substance,
                fingerprint_type,
                fingerprint_n_bits,
                verbose)
        except:
            print(f"WARNING: Unable to generate fingerprint for library molecule with index {substance_index}")
            continue
        fingerprints.append(fingerprint)

        substance_info = {}
        if fields is None:
            substance_info.update(substance.GetPropsAsDict())
        else:
            for field in fields:
                try:
                    field_value = substance.GetProp(field)
                    substance_info[field] = field_value
                except:
                    print(
                        f"WARNING: Library compound at index {substance_index} does not have field {field}.")
                    substance_info[field] = None
        substances.append(substance_info)

    if verbose:
        print(f"Found library contains {len(substances)} substances.")

    fingerprints = np.array(fingerprints)
    substances = pd.DataFrame(substances)

    return substances, fingerprints
