import torch
from scripts.compute_metrics import get_crystal_array_list, Crystal
import os
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter
import codecs
import smact
import smact.data_loader
from collections import Counter


def patch_smact_file_open():
    """Fix the file opening method for smact, enforcing UTF-8 encoding."""
    original_open = open
    def utf8_open(*args, **kwargs):
        kwargs['encoding'] = 'utf-8'
        return original_open(*args, **kwargs)
    smact.data_loader.open = utf8_open

# Define allowed element combinations
ALLOWED_ELEMENTS = {
    'A': ['Bi', 'Pb','Ca'],  # Allowed elements for A site
    'B': ['Fe', 'Ti'],  # Allowed elements for B site
    'X': ['O']          # Allowed elements for X site
}

def is_valid_composition(crystal):
    """Check if the structure contains the required element combinations."""
    elem_counter = Counter(crystal.atom_types)
    
    # Print the current composition for debugging
    print(f"Current composition: {dict(elem_counter)}")
    
    # Check if elements are within the allowed range
    for elem in elem_counter.keys():
        if not any(elem in allowed_list for allowed_list in ALLOWED_ELEMENTS.values()):
            return False
    
    # Check if it contains at least one A-site element, one B-site element, and oxygen
    has_A = any(elem in ALLOWED_ELEMENTS['A'] for elem in elem_counter.keys())
    has_B = any(elem in ALLOWED_ELEMENTS['B'] for elem in elem_counter.keys())
    has_X = any(elem in ALLOWED_ELEMENTS['X'] for elem in elem_counter.keys())
    
    return has_A and has_B and has_X

# Apply the patch
patch_smact_file_open()

# Get the current directory
current_directory = os.getcwd()
print("Current default path:", current_directory)

# Create the output directory for results
output_directory = os.path.join(current_directory, 'results', 'filtered_heat_ref')
os.makedirs(output_directory, exist_ok=True)
print(f"Results will be saved to: {output_directory}")

# Load the PT file, set weights_only=True (for improved security?)
def load_pt_file(file_path):
    try:
        return torch.load(file_path, weights_only=True)
    except Exception as e:
        print(f"Warning: Could not load with weights_only=True, trying without: {e}")
        return torch.load(file_path)

ptfile = 'cdvae/prop_models/perovskite/eval_recon_heat_ref.pt'
try:
    crys_array_list, _ = get_crystal_array_list(ptfile)
except Exception as e:
    print(f"Error loading crystal array list: {e}")
    raise

# Print information of the first structure for inspection
print("\nFirst crystal array info:")
try:
    crystal = Crystal(crys_array_list[0])
    print("Lengths:", crystal.lengths)
    print("Angles:", crystal.angles)
    print("Atom types:", crystal.atom_types)
    print("Fractional coordinates:", crystal.frac_coords)
    if hasattr(crystal, 'constructed'):
        print("Structure constructed:", crystal.constructed)
        if not crystal.constructed:
            print("Invalid reason:", crystal.invalid_reason)
except Exception as e:
    print(f"Error creating first crystal: {e}")
    raise

# Convert and save structure files
valid_structures = 0
total_structures = len(crys_array_list)

for idx, crys_array in enumerate(crys_array_list):
    try:
        crystal = Crystal(crys_array)
        if not crystal.constructed:
            print(f"Structure {idx} construction failed: {crystal.invalid_reason}")
            continue
        
        # Check if the composition is valid
        if not is_valid_composition(crystal):
            print(f"Structure {idx} composition not valid")
            continue
        
        # Convert to POSCAR format
        poscar = Poscar(crystal.structure)
        poscar.write_file(os.path.join(output_directory, f'structure_{valid_structures}.vasp'))
        
        # Save as CIF file
        cif = CifWriter(crystal.structure)
        cif.write_file(os.path.join(output_directory, f'structure_{valid_structures}.cif'))
        
        valid_structures += 1
        print(f"Saved valid structure {valid_structures}")
        
    except Exception as e:
        print(f"Error processing structure {idx}: {e}")

print(f"\nProcessing complete:")
print(f"Total structures: {total_structures}")
print(f"Valid structures: {valid_structures}")