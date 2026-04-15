#!/usr/bin/env python3

"""
Use T5 embeddings (from a numpy file) to construct a UMAP, save it and
save the reduced values of the embeddings.

Differences with respect to the original:

- Reads the T5 embeddings from a npz file, and the associated ids and KOs
- Many input files with embeddings can be given, and they are positional
  arguments, instead of giving a directory with the -e argument (which
  used to glob all the h5 files there)
- An optional --valid-ids file can be given to fiter out ids not in there
- There are --out-* arguments to optionally give name to the output files
"""

from datetime import datetime
import resource
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter as fmt

import numpy as np
import umap
import joblib  # to store the umap object


def main():
    args = get_args()

    ids, t5_embeddings, kos = load_embeddings(args.embeddings)

    if args.valid_ids:
        ids, t5_embeddings, kos = filter_valid(ids, t5_embeddings, kos,
                                               args.valid_ids)

    log('Performing UMAP dimension reduction...')
    reducer = umap.UMAP(n_components=args.ncomp, metric='euclidean',
                        verbose=args.verbose)
    umap_embeddings = reducer.fit_transform(t5_embeddings)

    log(f'Saving UMAP model to {args.out_model}')
    joblib.dump(reducer, args.out_model)

    log(f'Saving UMAP embeddings to {args.out_embed}')
    np.savez(args.out_embed,
             ids=ids,
             umap_embeddings=umap_embeddings,
             kos=kos)


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
    add('--valid-ids', help='if given, only ids that appear in the file will be considered')
    add('--out-model', default='umap_model.pkl', help='output file with the UMAP model')
    add('--out-embed', default='umap_embeddings.npz', help='output file with UMAP embeddings')
    add('-n', '--ncomp', type=int, default=2, help='number of UMAP dimensions')
    add('-v', '--verbose', action='store_true', help='be more verbose')
    return parser.parse_args()


def load_embeddings(embedding_files):
    """Return the ids, T5 embeddings and KOs from the given embedding files."""
    # Get the arrays from the first file, so we start with the correct types.
    fname = embedding_files[0]
    log('Loading T5 embeddings from:', fname)
    data = np.load(fname)
    ids = data['ids']
    ems = data['t5_embeddings']
    kos = data['kos']

    # Extend the arrays with the data from the rest of the files.
    for fname in embedding_files[1:]:
        log('Adding T5 embeddings from:', fname)
        data = np.load(fname)
        ids = np.append(ids, data['ids'], axis=0)
        ems = np.append(ems, data['t5_embeddings'], axis=0)
        kos = np.append(kos, data['kos'], axis=0)

    log('Number of loaded T5 embeddings:', len(ems))

    return ids, ems, kos


def filter_valid(ids, t5_embeddings, kos, valid_ids_file):
    """Return arrays filtered by the valid ids given in the file."""
    log('Loading valid ids from:', valid_ids_file)
    valid_ids = {line.split()[0] for line in open(valid_ids_file)}
    log(f'Number of loaded valid ids:', len(valid_ids))

    indices = [i for i, pid in enumerate(ids) if pid in valid_ids]
    log('Number of embeddings after filtering will be:', len(indices))

    return ids[indices], t5_embeddings[indices], kos[indices]



if __name__ == '__main__':
    main()
