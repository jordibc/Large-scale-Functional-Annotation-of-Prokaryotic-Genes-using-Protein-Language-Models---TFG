#!/usr/bin/env python3

"""
Extract metadata lines where the first column matches a protein ID
in the FASTA file.

The "metadata file" comes from KEGG and has contents like:

eco:b0001       K08278  21.0    Leader_Thr
eco:b0002       K12524  820.0   Homoserine_dh AA_kinase ACT_9 NAD_binding_3 ACT_7 ACT DUF6247
eco:b0003       K00872  310.0   GHMP_kinases_N GHMP_kinases_C MVD-like_N GAP1-C
eco:b0004       K01733  428.0   PALP Thr_synth_N
eco:b0006       K09861  258.0   H2O2_YaaD DUF6884

where "eco:b0001" and so on are protein ids, and this program
creates a new "metadata file" that looks the same but only includes
lines with proteins mentioned in the given fasta file.
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter


def main():
    args = get_args()

    pids = set()  # protein ids
    for line in open(args.fasta):
        if line.startswith('>'):
            pids.add(line[1:].split()[0])  # '>eco:b0001  thrL; ...' -> 'eco:b0001'
    print(f'We found {len(pids)} protein ids we are interested in.')

    nlines = 0  # number of lines in the kos file
    nmatch = 0  # number of lines with a protein id matching one in pids
    with open(args.output, 'wt') as fout:
        for line in open(args.kos):  # will filter lines of the kos file
            nlines += 1
            if line.split()[0] in pids:  # 'eco:b0001       K08278 ...' -> 'eco:b0001'
                fout.write(line)
                nmatch += 1

    if nmatch > 0:
        print(f'Created {args.output} with {nmatch} entries (of {nlines}) from {args.kos}')
    else:
        print(f'No matching sequences found between {args.fasta} and {args.kos}')


def get_args():
    """Return the command-line arguments."""
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawDescriptionHelpFormatter)
    add = parser.add_argument  # shortcut
    add('fasta', help='FASTA file with the proteins whose ids we want')
    add('kos', help='KEGG file with protein ids and their KOs')
    add('output', help='file that will contain the filtered KEGG entries')
    return parser.parse_args()


if __name__ == '__main__':
    main()
