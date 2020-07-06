# -*- tab-width:4;indent-tabs-mode:nil;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=2 noet:

import os
import logging
from typing import Sequence
import rdkit
import rdkit.Chem
import rdkit.Chem.rdMolDescriptors

def library_search(
        query: Sequence[str],
        query_ids: Sequence[str],
        library_path: str,
        library_fields: Sequence[str],
        fingerprint_type: str = 'ECFP4',
        similarity_threshold: float = 0.6,
        verbose=False):
    """
    Screen a set of query molecules against a library for chemical similarity

    Args:
        query (Sequence[str]): a sequence of SMILES for each query compound
        query_ids: a sequence of identifiers for each query compound
        libray: path to library .sdf file
        library_fields: fields in the library .sdf file to be reported in the results
        fingerprint_type: type of fingerprint to represent molecules for comparison

    Returns:
        List[Dict[query_id:str, <library_fields>, tanimoto_similarity:float]]
    """

    # validate input
    if len(query) != len(query_ids):
        raise ValueError(f"Input query and query_ids lengths don't match. query has length {len(query)}, while query_ids has length {len(query_ids)}")

    if not os.path.exists(library_path):
        raise ValueError(f"The library path '{library_path}' does not exist.")

    valid_fingerprint_types = ['ECFP4']
    if fingerprint_type not in valid_fingerprint_types:
        raise ValueError((
            f"Unrecognized fingerprint_type '{fingerprint_type}'. ",
            f"Valid options are [{', '.join(valid_fingerprint_types)}]"))

    if similarity_threshold < 0 or 1 < similarity_threshold:
        raise ValueError((
            f"Similarity threshold {similarity_threshold} is not in the range [0, 1]"))

    query_fingerprints = []
    validated_query_ids = []
    for query_index, query_smiles in enumerate(query):
        query_molecule = rdkit.Chem.MolFromSmiles(query_smiles, sanitize=False)
        if query_molecule is None:
            print((
                f"WARNING: RDKit failed to create molecule '{query_ids[query_index]}' ",
                f"with smiles '{query_smiles}'; skipping..."))
            continue

        try:
            query_molecule = rdkit.Chem.RemoveHs(query_molecule) # Also Sanitizes
        except ValueError as e:
            print((
                f"WARNING: {str(e)}. Skipping molecule '{query_ids[query_index]}' ",
                f"with smiles '{query_smiles}'."))
            continue

        if fingerprint_type == "ECFP4":
            query_fingerprint = rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol=query_molecule, radius=2)
            query_fingerprints.append(query_fingerprint)
            validated_query_ids.append(query_ids[query_index])

    results = []
    for library_substance_index, library_substance in enumerate(rdkit.Chem.SDMolSupplier(library_path)):
        if fingerprint_type == "ECFP4":
            try:
                library_fingerprint = rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol=library_substance, radius=2, nBits=1024)
            except:
                print(f"WARNING: Unable to generate fingerprint for library molecule with index {library_substance_index}")
                continue

        for query_index, query_fingerprint in enumerate(query_fingerprints):
            tanimoto_similarity = rdkit.DataStructs.FingerprintSimilarity(
                fp1=query_fingerprint,
                fp2=library_fingerprint)
            if tanimoto_similarity >= similarity_threshold:
                if verbose:
                    print(f"For query {query_ids[query_index]} found similar library compound at index {library_substance_index}")
                result = {'query_id' : query_ids[query_index]}
                if library_fields is None:
                    result.update(library_substance.GetPropsAsDict())
                else:
                    for library_field in library_fields:
                        try:
                            library_field_value = library_substance.GetProp(library_field)
                            result[library_field] = library_field_value
                        except:
                            print(
                                f"WARNING: Library compound at index {library_substance_index} does not have field {library_field}.")
                            result[library_field] = None
                result['tanimoto_similarity'] = tanimoto_similarity
                results.append(result)

    if verbose:
        print(f"Found {len(results)} similar hits.")

    return results
