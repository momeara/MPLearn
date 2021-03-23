# -*- tab-width:4;indent-tabs-mode:nil;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=2 noet:


def generate_fingerprints_smiles(
        smiles: str,
        fingerprint_type: str = 'ECFP4',
        verbose=False):
    """
    Generate fingerprints for a set of molecules

    Args:
        smiles: a list of smiles strings
        fingerprint_type: type of fingerprint to represent molecules for comparison

    Returns:
        List[Dict[query_id:str, <library_fields>, tanimoto_similarity:float]]
    """
    valid_fingerprint_types = ['ECFP4']
    if fingerprint_type not in valid_fingerprint_types:
        raise ValueError((
            f"Unrecognized fingerprint_type '{fingerprint_type}'. ",
            f"Valid options are [{', '.join(valid_fingerprint_types)}]"))

    if isinstance(substance, str):
        substance = [substance]

    fingerprints = []
    for index, substance_smiles in enumerate(smiles)
        if fingerprint_type == "ECFP4":
            molecule = rdkit.Chem.MolFromSmiles(substance_smiles, sanitize=False)
            if molecule is None:
                print((
                    f"WARNING: RDKit failed to create molecule '{index}' ",
                    f"with smiles '{substance_smiles}'; skipping..."))
                fingerprints.append(None)
                continue

                try:
                    molecule = rdkit.Chem.RemoveHs(molecule) # Also Sanitizes
                except ValueError as e:
                    print((
                        f"WARNING: {str(e)}. Skipping molecule '{index}' ",
                        f"with smiles '{smiles}'."))
                    fingerprints.append(None)
                    continue
            try:
                fingerprint = rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol=molecule, radius=2, nBits=1024)

            except:
                print(f"WARNING: Unable to generate fingerprint for library molecule with index {index}")
                fingerprints.append(None)
                continue
            fingerprints.append(fingerprint)
    fingerprints = np.array(fingerprints)
    return fingerprints
    

def generate_fingerprints_sdf(
        library_path: str,
        library_fields: Sequence[str],
        fingerprint_type: str = 'ECFP4',
        verbose=False):
        
    """
    Generate fingerprints for substances in the library

    Args:
        libray: path to library .sdf file
        library_fields: fields in the library .sdf file to be reported in the results
        fingerprint_type: type of fingerprint to represent molecules for comparison

    Returns:
        List[Dict[query_id:str, <library_fields>, tanimoto_similarity:float]]
    """

    # validate inputs
    if not os.path.exists(library_path):
        raise ValueError(f"The library path '{library_path}' does not exist.")

    valid_fingerprint_types = ['ECFP4']
    if fingerprint_type not in valid_fingerprint_types:
        raise ValueError((
            f"Unrecognized fingerprint_type '{fingerprint_type}'. ",
            f"Valid options are [{', '.join(valid_fingerprint_types)}]"))

    substances = []
    fingerprints = []
    for library_substance_index, library_substance in enumerate(rdkit.Chem.SDMolSupplier(library_path)):
        if fingerprint_type == "ECFP4":
            try:
                import pdb; pdb.set_trace()
                library_fingerprint = rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol=library_substance, radius=2, nBits=1024)
            except:
                print(f"WARNING: Unable to generate fingerprint for library molecule with index {library_substance_index}")
                continue


        substance_info = {}
        if library_fields is None:
            substance_info.update(library_substance.GetPropsAsDict())
        else:
            for library_field in library_fields:
                try:
                    library_field_value = library_substance.GetProp(library_field)
                    substance_info[library_field] = library_field_value
                except:
                    print(
                        f"WARNING: Library compound at index {library_substance_index} does not have field {library_field}.")
                    substance_info[library_field] = None
        substance_info["fingerprint"] = library_fingerprint
        substance_info.append(result)

    if verbose:
        print(f"Found library contains {len(results)} substances.")

    fingerprint = np.array(fingerprints)
    substances = pd.DataFrame(substances)

    return fingerprints, substances
    
