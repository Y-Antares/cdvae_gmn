import pandas as pd
from pymatgen.core.structure import Structure

def analyze_atom_counts(dataset_path):
    df = pd.read_csv(dataset_path)
    atom_counts = []
    
    for _, row in df.iterrows():
        try:
            crystal = Structure.from_str(row['cif'], fmt='cif')
            atom_counts.append(len(crystal))
        except:
            pass
    
    print(f"最小原子数: {min(atom_counts)}")
    print(f"最大原子数: {max(atom_counts)}")
    print(f"平均原子数: {sum(atom_counts)/len(atom_counts):.2f}")
    
analyze_atom_counts("/root/cdvae/data/perov_5/train.csv")