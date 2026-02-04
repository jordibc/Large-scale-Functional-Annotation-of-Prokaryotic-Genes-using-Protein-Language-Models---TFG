#!/usr/bin/env python3

"""
Prepare KO labels for KOPNet training.
"""

import numpy as np
import argparse


def main():
    args = get_args()

    print(f"Loading protein ids and their ko terms from {args.metadata} ...")
    id2ko = get_id2ko(args.metadata)
    print(f"Loaded {len(id2ko)} protein ids with their ko terms.")

    print("Reading protein ids we saved in the umap file...")
    ids = np.load(args.umap_reduction, allow_pickle=True)['ids']

    kos = [id2ko.get(pid, 'NA') for pid in ids]  # kos corresponding to the protein ids

    missing = set(ids) - set(id2ko.keys())
    if missing:
        print('Warning: missing ids', missing)

    np.savetxt(f"{args.output}.txt", np.array(kos), fmt="%s")

    print(f"Ko annotations ready for training can be found in {args.output}")


def get_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-u", "--umap_reduction", required=True, help="Path to the input UMAP reduction")
    parser.add_argument("-m", "--metadata", required=True, help="Path to the input metadata")
    parser.add_argument("-o", "--output", required=True, help="Resulting file with ko prepared for training")
    return parser.parse_args()


def get_id2ko(metadata_file):
    """Return {id: ko} from file with KOs per sequence."""
    id_ko_dict = {}
    for line in open(metadata_file):
        prot_id, ko = line.strip().split("\t")[:2]

        norm_id = prot_id.replace(".", "_").rstrip().split(" ")[0]  # same format as embedding ids
        norm_ko = ko.replace("/", "_")  # multiple kos (Kxxx1/Kxxx2) go into a single class Kxxx1_Kxxx2

        id_ko_dict[norm_id] = norm_ko

    return id_ko_dict



if __name__ == "__main__":
    main()
