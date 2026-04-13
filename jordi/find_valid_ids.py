#!/usr/bin/env python

"""
Use the KEGG ".dat" file with KOs information to generate the following files:

- id_ko.txt: valid ids and their associated kos
- ko_id.txt: *all* kos and their associated ids

We'll use id_ko.txt to decide what to encode (with T5 and UMAP). It looks like:

  eco:b0001	K08278
  eco:b0002	K12524
  eco:b0003	K00872
  ...

We use ko_id.txt for reference only. It contains for each KO the associated
ids, sorted from the KOs that appear more often to less often. It looks like:

  ...
  K16186	loki:Lokiarch_05450	loki:Lokiarch_09660	psyt:DSAG12_00977
  K12188	loki:Lokiarch_37450	lob:NEF87_001464	psyt:DSAG12_03697
  K19150	eco:b4668	ebe:B21_04369
  ...
"""

from argparse import ArgumentParser, RawTextHelpFormatter as fmt
import gzip


def main():
    args = get_args()

    # Protein ids associated to each KO in the file.
    print('Reading ids and KOs from:', args.kos_file)

    ko2ids = {}
    for pid, ko in id_ko(args.kos_file, args.truncate):
        ko2ids.setdefault(ko, [])
        ko2ids[ko].append(pid)

    # Write id_ko.txt with only valid values.
    print('Writing:', args.id_out)

    with open(args.id_out, 'wt') as f:  # we loop again over the original file
        for pid, ko in id_ko(args.kos_file, args.truncate):
            if len(ko2ids[ko]) >= args.min:  # valid?
                f.write(f'{pid}\t{ko}\n')  # only write the valid values!

    # Write ko_id.txt with KOs sorted from more present to less, and their ids.
    print('Writing:', args.ko_out)

    with open(args.ko_out, 'wt') as f:
        for ko, ids in sorted(ko2ids.items(),
                              key=lambda kv: len(kv[1]), reverse=True):
            f.write(ko + '\t' + '\t'.join(ids) + '\n')


def get_args():
    """Return the command-line arguments."""
    parser = ArgumentParser(description=__doc__, formatter_class=fmt)
    add = parser.add_argument  # shortcut

    # Defaults.
    kos_file = '/home/huerta/_Databases/kegg.07-24/genes/fasta/prokaryotes.dat.gz'
    id_out = 'id_ko.txt'
    ko_out = 'ko_id.txt'

    add('--kos-file', default=kos_file, help=f'file with protein ids and their KOs (default: {kos_file})')
    add('--id-out', default=id_out, help=f'file that will have the valid ids and associated KOs (default: {id_out})')
    add('--ko-out', default=ko_out, help=f'file that will have all KOs and their associated ids (default: {ko_out})')
    add('--min', type=int, default=2, help=f'minimum number of associated ids for a KO to be considered valid (default: 2)')
    add('--truncate', type=int, help='if specified, use only the first given number of lines from the KOs file (for tests)')

    return parser.parse_args()


def id_ko(fname, truncate):
    """Yield pairs of (id, ko) from the given filename."""
    for i, line in enumerate(zopen(fname)):
        if truncate and i > truncate:
            return

        parts = line.rstrip('\n').split('\t')

        if len(parts) > 1:
            pid, ko = parts[:2]
            if ko:
                yield pid, ko


def zopen(fname):
    """Return the file object related to the given (possibly gzipped) file."""
    # We could test like this too: open(fname, 'rb').read(2) == b'\x1f\x8b'
    return gzip.open(fname, 'rt') if fname.endswith('.gz') else open(fname)



if __name__ == '__main__':
    main()
