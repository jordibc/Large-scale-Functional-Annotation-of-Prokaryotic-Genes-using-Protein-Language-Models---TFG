#!/usr/bin/env python

"""
Use embeddings to construct a UMAP, save it and save the reduced values of the embeddings.
"""

from datetime import datetime
import glob
import resource
import gc
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter as fmt

from tqdm import tqdm
import umap
import numpy as np
import h5py
import joblib  # to store the umap object


def main():
    args = get_args()

    log('Loading protein ids with associated kos...')
    valid_ids = set(line.split()[0].replace('.', '_') for line in open(args.kos_file))

    log('Loading embeddings...')
    embeddings, prot_ids = load_embeddings_for_umap(args.embeddings)

    log('Filtering embeddings of proteins with associated ko...')
    indices = [i for i, pid in enumerate(prot_ids) if pid in valid_ids]
    filtered_ids = [prot_ids[i] for i in indices]
    filtered_embeddings = embeddings[indices]

    log('Clearing original embeddings from memory...')
    del embeddings
    gc.collect()

    log('Performing UMAP dimension reduction...')
    reducer = umap.UMAP(n_components=args.ncomp, metric='euclidean', verbose=args.verbose)
    emb_umap = reducer.fit_transform(filtered_embeddings)

    log('Saving umap model...')
    joblib.dump(reducer, 'umap_model.pkl')

    log('Saving umap embeddings...')
    np.savez('umap_embeddings.npz',
             ids=filtered_ids,  # protein ids
             coordinates=emb_umap)  # umap embeddings

    log('End of script')  # includes the total execution time


t0 = datetime.now()

def log(*args):
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    dt = datetime.now() - t0
    print(f'[{dt}] [{round(mem,2)} GB]', *args, flush=True)


def get_args():
    """Return the parsed command line arguments."""
    parser = ArgumentParser(description=__doc__, formatter_class=fmt)
    add = parser.add_argument

    add('embeddings',
        help='path to directory storing embeddings (.h5)')

    add('kos_file',
        help='file with protein ids and their kos, to only use proteins with kos')

    add('-n', '--ncomp', type=int, default=2,
        help='number of components for UMAP dimensionality reduction')

    add('-v', '--verbose', action='store_true')

    return parser.parse_args()


###############################################################################
#                 GET EMBEDDINGS FROM H5 FILES                                #
###############################################################################

def load_embeddings_for_umap(embeddings_path):
    """
    Loads all embeddings contained in embeddings_path

    Args:
        embeddings_path: path to directory containing .h5 files with protein embeddings

    Returns:
        Tuple with (embeddings_array, ids)
    """
    # Create external links
    table_link = 'table_links.h5'

    h5_files = sorted(glob.glob(f'{embeddings_path}/*.h5'))
    log(f'Found {len(h5_files)} H5 files in {embeddings_path}')

    with h5py.File(table_link, mode='w') as h5fw:
        for i, h5name in enumerate(h5_files, 1):
            h5fw[f'link{i}'] = h5py.ExternalLink(h5name, '/')

    log(f'Linked h5 files: {len(h5_files)}')

    # First round: get embedding dimension and number to estimate needed memory
    embedding_dim = None
    total_embeddings = 0

    log('Counting embeddings and getting dimensions...')
    with h5py.File('table_links.h5', 'r') as myfile:
        for group_name in tqdm(myfile.keys(), desc="Scanning files"):
            if isinstance(myfile[group_name], h5py.Group):
                group = myfile[group_name]

                for dataset_name in group.keys():
                    # Read embedding's shape
                    if embedding_dim is None:
                        embedding_shape = group[dataset_name].shape
                        embedding_dim = embedding_shape[0]

                    total_embeddings += 1

    log(f'Embeddings dimension: {embedding_dim}')
    log(f'Number of embeddings to load: {total_embeddings}')

    # Preallocate array to save embeddings
    # Using float32 instead of float64 to save memory if precision is sufficient
    all_embeddings = np.zeros((total_embeddings, embedding_dim), dtype=np.float32)
    all_ids = []

    # Second round: load embeddings
    log('Loading embeddings...')
    current_index = 0

    with h5py.File('table_links.h5', 'r') as myfile:
        # Process groups in sorted order for reproducibility
        for group_name in tqdm(sorted(myfile.keys()), desc="Loading embeddings"):
            if isinstance(myfile[group_name], h5py.Group):
                group = myfile[group_name]

                # Prepare dataset list first to minimize IO operations
                datasets_to_load = list(group.keys())

                # Load embeddings of current group
                for dataset_name in datasets_to_load:
                    dataset_id = dataset_name.replace(".", "_").rstrip().split(" ")[0]
                    all_embeddings[current_index] = group[dataset_name][()]
                    all_ids.append(dataset_id)
                    current_index += 1

                # Release memory after processing each group
                gc.collect()

    log(f'Loaded embeddings: {current_index}/{total_embeddings}')

    return all_embeddings, all_ids



if __name__ == "__main__":
    main()
