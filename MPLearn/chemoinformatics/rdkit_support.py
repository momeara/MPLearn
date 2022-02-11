
import os
from rdkit.Chem.rdmolfiles import MolFromMol2Block



# adapted from: https://chem-workflows.com/articles/2019/07/18/building-a-multi-molecule-mol2-reader-for-rdkit/
def Mol2MolSupplier(file,sanitize=True):
    m = None
    line_index=0
    start_line=None
    with open(file, 'r') as f:
        line =f.readline()
        line_index += 1
        # skip down to the beginning of the first molecule
        while not line.startswith("@<TRIPOS>MOLECULE") and not f.tell() == os.fstat(f.fileno()).st_size:
            line = f.readline()
            line_index += 1
        while not f.tell() == os.fstat(f.fileno()).st_size:
            if line.startswith("@<TRIPOS>MOLECULE"):
                mol = []
                mol.append(line)
                start_line = line_index
                line = f.readline()
                line_index += 1
                while not line.startswith("@<TRIPOS>MOLECULE"):
                    mol.append(line)
                    line = f.readline()
                    line_index += 1
                    if f.tell() == os.fstat(f.fileno()).st_size:
                        mol.append(line)
                        break
                mol[-1] = mol[-1].rstrip() # removes blank line at file end
                block = ",".join(mol).replace(',','') + "\n"
                m=MolFromMol2Block(block,sanitize=sanitize)
            yield (start_line, m)
