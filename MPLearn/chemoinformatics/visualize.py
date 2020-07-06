# -*- tab-width:4;indent-tabs-mode:nil;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=2 noet:


import rdkit
import rdkit.Chem
import rdkit.Chem.rdFMCS
import rdkit.Chem.AllChem
import rdkit.Chem.Draw

def draw_aligned_substances(
        substance_smiles,
        substance_ids,
        template_smiles=None,
        verbose=False):

    if len(substance_smiles) != len(substance_ids):
        raise ValueError(f"Input substance_smiles and substance_ids lengths don't match. substance_smiles has length {len(substance_smiles)}, while substance_ids has length {len(substance_ids)}")

    rdkit.Chem.rdDepictor.SetPreferCoordGen(True)
    
    substances = []
    for substance_index, smiles in enumerate(substance_smiles):
        try:
            substance = rdkit.Chem.MolFromSmiles(smiles, sanitize=False)            
        except:
            print((
                f"WARNING: RDKit failed to create substance '{substance_ids[substance_index]}' ",
                f"with smiles '{smiles}'; skipping..."))
            continue
            
        if substance is None:
            print((
                f"WARNING: RDKit failed to create substance '{substance_ids[substance_index]}' ",
                f"with smiles '{smiles}'; skipping..."))
            continue

        try:
            query_molecule = rdkit.Chem.RemoveHs(substance) # Also Sanitizes
        except ValueError as e:
            print((
                f"WARNING: {str(e)}. Skipping substance '{substance_ids[substance_index]}' ",
                f"with smiles '{smiles}'."))
            continue
        substance.UpdatePropertyCache()
        rdkit.Chem.SetHybridization(substance)
        rdkit.Chem.Kekulize(substance)
        rdkit.Chem.AllChem.Compute2DCoords(substance)

        
        substances.append(substance)

    if template_smiles is None:
        template = rdkit.Chem.rdFMCS.FindMCS(substances)
        template = rdkit.Chem.MolFromSmarts(template.smartsString)
    else:
        template = rdkit.Chem.MolFromSmiles(template_smiles, sanitize=False)
        if substance is None:
            print((
                f"ERROR: RDKit failed to create template from smiles '{template_smiles}'; skipping..."))
            exit(1)

        try:
            template_molecule = rdkit.Chem.RemoveHs(template) # Also Sanitizes
        except ValueError as e:
            print((
                f"ERROR: {str(e)} RDKit failed to sanitize template with from smiles '{template_smiles}'."))
            exit(1)
    template.UpdatePropertyCache()
    rdkit.Chem.SetHybridization(template)
    rdkit.Chem.AllChem.Compute2DCoords(template)


    rdkit.Chem.Draw.DrawingOptions.atomLabelFontSize = 55
    rdkit.Chem.Draw.DrawingOptions.dotsPerAngstrom = 100
    rdkit.Chem.Draw.DrawingOptions.bondLineWidth = 3.0
    
    depictions = []
    for substance in substances:
        
        rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(substance, template)
        image = rdkit.Chem.Draw.MolToImage(
            substance,
            size=(1000, 1000))
        depictions.append(image)
    #import pdb; pdb.set_trace()
    #z = rdkit.Chem.Draw.MolsToGridImage(substances, molsPerRow=5, subImgSize=(500,500), useSVG=True)
    #z.save('product/figures/z.pdf')
    return depictions
