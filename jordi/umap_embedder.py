#!/usr/bin/env python

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
import gc
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter as fmt
import gzip

import numpy as np
import umap
import joblib  # to store the umap object


def main():
    args = get_args()

    log('Loading T5 embeddings...')
    ids, t5_embeddings, kos = load_embeddings(args.embeddings)
    log(f'Loaded {len(t5_embeddings)} T5 embeddings.')


    if args.valid_ids:
        valids = {line.split()[0] for line in open(args.valid_ids)}
        print(f'Using {len(valids)} ids from {args.valid_ids}')

        indices = [i for i, pid in enumerate(ids) if pid in valids]
        ids = ids[indices]
        t5_embeddings = t5_embeddings[indices]
        kos = kos[indices]
        log(f'{len(indices)} remain.')

    log('Performing UMAP dimension reduction...')
    reducer = umap.UMAP(n_components=args.ncomp, metric='euclidean',
                        verbose=args.verbose)
    umap_embeddings = reducer.fit_transform(t5_embeddings)

    out_model = args.out_model or 'umap_model.pkl'
    log(f'Saving UMAP model to {out_model}')
    joblib.dump(reducer, out_model)

    out_embed = args.out_embed or 'umap_embeddings.npz'
    log(f'Saving UMAP embeddings to {out_embed}')
    np.savez(out_embed,
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


def zopen(fname):
    """Return the file object related to the given (possibly gzipped) file."""
    # We could test like this too: open(fname, 'rb').read(2) == b'\x1f\x8b'
    return gzip.open(fname, 'rt') if fname.endswith('.gz') else open(fname)


def load_embeddings(embedding_files):
    """Return the ids and embeddings from the given list of embedding files."""
    data = np.load(embedding_files[0])  # the first one gives us the correct types
    ids = [normalize(sid) for sid in data['ids']]
    t5_embeddings = data['t5_embeddings']
    kos = data['kos']

    for f in embedding_files[1:]:  # for the other files we just extend the arrays
        data = np.load(f)
        ids = np.concat([ids, [normalize(sid) for sid in data['ids']]])
        t5_embeddings = np.concat([t5_embeddings, data['t5_embeddings']])
        kos = np.concat([kos, data['kos']])

    return ids, t5_embeddings, kos


def normalize(pid):
    """Return a protein id "normalized" as it appears in the kos file."""
    return pid.split()[0].replace('.', '_')  # 'eco:b0001  thrL; ...' -> 'eco:b0001'



if __name__ == "__main__":
    main()
