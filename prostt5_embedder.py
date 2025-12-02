#!/usr/bin/env python3

"""
Create T5 embeddings for a given (multi)fasta file.
"""

# Originally created on Wed Sep 23 18:33:22 2020 by mheinzinger.
# Edited on Dec 2025 by JBC.

from argparse import ArgumentParser

import torch
import h5py
from transformers import T5EncoderModel, T5Tokenizer


def main():
    args = get_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    emb_dict = get_embeddings(args.input, args.model, device=device,
                              per_protein=(args.per_protein != 0),
                              max_residues=args.max_residues,
                              max_seq_len=args.max_seq_len,
                              max_batch=args.max_batch)

    with h5py.File(args.output, 'w') as hf:
        for sequence_id, embedding in emb_dict.items():
            hf.create_dataset(sequence_id, data=embedding)

    print(f'Created file "{args.output}" with {len(emb_dict)} embedding(s).')


def get_args():
    """Return the command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    add = parser.add_argument  # shortcut

    add('-i', '--input', required=True, help='fasta file with protein sequence(s)')
    add('-o', '--output', required=True, help='output file with the created embeddings in h5 format')

    add('--model', help='path to a checkpoint of the pre-trained model')
    add('--per_protein', type=int, default=0, choices=[0, 1],
        help='return per-residue embeddings (0, default), or mean-pooled per-protein representation (1)')
    # Would better have been this, but I am not changing it for compatibility:
    #   add('--per-protein', action='store_true',
    #       help='return mean-pooled per-protein representation, instead of per-residue embeddings')

    add('--max-residues', type=int, default=4000, help='number of cumulative residues per batch')
    add('--max-batch', type=int, default=100, help='max number of sequences per single batch')
    add('--max-seq-len', type=int, default=1000,
        help='max length after which we switch to single-sequence processing to avoid OOM')

    return parser.parse_args()


def get_embeddings(seq_path, model_dir, device,
                   per_protein, # whether to derive per-protein (mean-pooled) embeddings
                   max_residues=4000, # number of cumulative residues per batch
                   max_seq_len=1000, # max length after which we switch to single-sequence processing to avoid OOM
                   max_batch=100, # max number of sequences per single batch
                   verbose=True):
    seq_dict = read_fasta(seq_path)
    print(f'Read {len(seq_dict)} sequences from {seq_path}.')

    model, vocab = get_T5_model(model_dir, device)

    avg_length = sum(len(seq) for seq in seq_dict.values()) / len(seq_dict)
    n_long = sum(1 for seq in seq_dict.values() if len(seq) > max_seq_len)
    sequences_sorted = sorted(seq_dict.items(), key=lambda id_seq: len(id_seq[1]), reverse=True)
    # We will iterate on the sequences from longer to shorter... in case to fail early?

    if verbose:
        print(f'Average sequence length: {avg_length}')
        print(f'Number of sequences longer than {max_seq_len}: {n_long}')

    emb_dict = dict()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(sequences_sorted, 1):
        seq = seq.replace('U','X').replace('Z','X').replace('O','X')
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs, add_special_tokens=True, padding='longest')
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print('RuntimeError during embedding for {} (L={}). Try lowering batch size. '.format(pdb_id, seq_len) +
                      'If single sequence processing does not work, you need more vRAM to process your protein.')
                continue

            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # slice-off padded/special tokens
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]

                if per_protein:
                    emb = emb.mean(dim=0)

                if verbose:
                    print(f'Embedded protein "{identifier}" with length {s_len} into {tuple(emb.shape)} dimensions.')

                emb_dict[identifier] = emb.detach().cpu().numpy().squeeze()

    return emb_dict


def read_fasta(fname):
    """Return a dict  d[name] = seq  with the contents of a fasta file."""
    seqs = {}

    for line in open(fname):
        line = line.rstrip()  # remove trailing whitespace

        if not line or line.startswith(';'):
            pass
        elif line.startswith('>'):
            name = line[1:].replace('/', '_').replace('.', '_')
            # The replacements are to avoid misinterpretations when loading h5.
            seqs[name] = ''
        else:
            seqs[name] += line.upper().replace('-', '')
            # Drop gaps and cast to upper-case, as in the original prostt5_embedder.py.

    return seqs


def get_T5_model(model_dir, device, transformer_link='Rostlab/ProstT5'):
    print(f'Loading: {transformer_link}')
    if model_dir is not None:
        print(f'Loading cached model from: {model_dir}')

    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)

    # Only cast to full-precision if no GPU is available.
    if not torch.cuda.is_available():
        print('Casting model to full precision for running on CPU...')
        model.to(torch.float32)

    model = model.to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, legacy=True)

    return model, vocab



if __name__ == '__main__':
    main()
