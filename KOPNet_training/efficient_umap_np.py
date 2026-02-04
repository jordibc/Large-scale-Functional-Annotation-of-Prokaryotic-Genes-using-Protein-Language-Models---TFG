#!/usr/bin/env python

"""
Use T5 embeddings (from a numpy file) to construct a UMAP, save it and
save the reduced values of the embeddings.

Differences with respect to the original:

- Reads the T5 embeddings from a npz file
- Many input files with embeddings can be given, and they are positional
  arguments, instead of giving a directory with the -e argument (which
  used to glob all the h5 files there)
- The last positional argument is the file with protein ids and their kos,
  used to filter out the ids for which we don't have a ko
"""

from datetime import datetime
import resource
import gc
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter as fmt

import numpy as np
import umap
import joblib  # to store the umap object


def main():
    args = get_args()

    log('Loading protein ids with associated kos...')
    valid_ids = set(line.split()[0] for line in open(args.kos_file))

    log('Loading embeddings...')
    seq_ids, embeddings = load_embeddings(args.embeddings)
    log(f'Loaded {len(embeddings)} embeddings')

    log('Filtering embeddings of proteins with associated ko...')
    indices = [i for i, pid in enumerate(seq_ids) if pid in valid_ids]
    filtered_ids = [seq_ids[i] for i in indices]
    filtered_embeddings = embeddings[indices]
    log(f'{len(filtered_embeddings)} remain')

    log('Clearing original embeddings from memory...')  # really necessary?
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
    add('embeddings', nargs='+', help='file with T5 embeddings (in npz format)')
    add('kos_file', help='file with protein ids and their kos, to only use proteins with kos')
    add('-n', '--ncomp', type=int, default=2, help='number of components for UMAP dimensionality reduction')
    add('-v', '--verbose', action='store_true')
    return parser.parse_args()


def load_embeddings(embeddings):
    """Return the ids and embeddings from the given list of embedding files."""
    data = np.load(embeddings[0])  # the first one gives us the correct types
    seq_ids = [normalize(sid) for sid in data['seq_ids']]
    embeddings = data['embeddings']

    for f in embeddings[1:]:  # for the other files we just extend the arrays
        data = np.load(f)
        seq_ids = np.concat([seq_ids, [normalize(sid) for sid in data['seq_ids']]])
        embeddings = np.concat([embeddings, data['embeddings']])

    return seq_ids, embeddings


def normalize(pid):
    """Return a protein id "normalized" as it appears in the kos file."""
    return pid.split()[0].replace('.', '_')  # 'eco:b0001  thrL; ...' -> 'eco:b0001'



if __name__ == "__main__":
    main()
