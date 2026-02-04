# PREPARE KO LABELS FOR TRAINING


## DEPENDENCIES
import numpy as np
import argparse
print("Dependencies loaded")


## FUNCTIONS

### Parse arguments from terminal
def parse_args():
    '''Parse argument from terminal.'''
    # Create parser
    parser = argparse.ArgumentParser(
        description="Perpare KO labels for KOPNet training"
    )
    # Add arguments
    parser.add_argument("-u", "--umap_reduction", required=True, help="Path to the input UMAP reduction")
    parser.add_argument("-m", "--metadata", required=True, help="Path to the input metadata")
    parser.add_argument("-o", "--output", required=True, help="Resulting file with ko prepared for training")
    # Parse file from command line
    args = parser.parse_args()
    return args

### Normalize protein id
def normalize_id(prot_id):
    '''Returns protein id with same format as embeddings ids'''
    return prot_id.replace(".", "_").rstrip().split(" ")[0]

### Normalize multiple KO annotation
def normalize_multiple_ko(ko):
    '''Transform multiple ko annotations (Kxxx1/Kxxx2) into single class Kxxx1_Kxxx2'''
    return ko.replace("/", "_")

### Read metadata file
def build_id_ko_dict(metadata_file):
    '''Read metadata file including KO id per sequence intended
    to be used in training. Returns dict {'id':'ko'}'''
    # Initialize dict
    id_ko_dict={}
    # Read metadata file & store in dict
    with open(metadata_file, 'r') as mf:
        for protein in mf:
            prot_id, ko = protein.strip().split("\t")[:2]
            norm_id = normalize_id(prot_id)
            norm_ko=normalize_multiple_ko(ko)
            id_ko_dict[norm_id] = norm_ko
    return id_ko_dict

### Load UMAP coordinates and protein ids
def load_umap_data(umap_red):
    '''Extracts umap coordinates and id for each embedding'''
    data=np.load(umap_red, allow_pickle=True)
    ids = data['ids']
    coordinates = data['coordinates']
    return ids, coordinates

            

### MAIN EXECUTION

if __name__ == "__main__":

    # Parse command line arguments
    args = parse_args()
    umap_red = args.umap_reduction
    metadata_file = args.metadata
    output = args.output
    print(f"UMAP reduction file: {umap_red}")
    print(f"Metadata: {metadata_file}")

    # Build dict storing id-ko equivalence
    id_ko_dict=build_id_ko_dict(metadata_file)
    print(f"Loaded {len(id_ko_dict.keys())} protein ids and their {len(id_ko_dict.values())} ko terms terms")

    # Load umap reduced coordinates of each embedding & embeddings ids
    embeddings_ids, umap_coordinates = load_umap_data(umap_red)
    print(f"Loaded {umap_coordinates.shape[0]} UMAP coordinates and {len(embeddings_ids)} embedding ids")

    # Align protein ids with embeddings ids
    aligned_kos = [id_ko_dict[e_id] if e_id in id_ko_dict else (print(f"Warning: '{e_id}' not found") or "NA") for e_id in embeddings_ids]

    # Save aligned ko annotation for model training
    np.savetxt(f"{output}.txt", np.array(aligned_kos), fmt="%s")  
    print(f"Ko annotations ready for training can be found in {output}")
    
