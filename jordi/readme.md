# Alternative versions

## Pipeline order

1. `find_valid_ids.py`
2. `t5_embedder.py`
3. `umap_embedder.py`
4. `kopnet.py`

Their input and output files are:

| program             | input                       | output                                     |
| ------------------- | --------------------------- | ------------------------------------------ |
| `find_valid_ids.py` | `dat` (KOs)                 | `id_ko.txt` (valid ids and KOs)            |
| `t5_embedder.py`    | `pep` (fastas), `id_ko.txt` | `t5_embeddings.npz` (includes ids and KOs) |
| `umap_embedder.py`  | `t5_embeddings.npz`         | `umap_embeddings.npz`, `umap_model.pkl`    |
| `kopnet.py`         | `umap_embeddings.npz`       | `model.pt`                                 |

All the programs accept the `--help` flag to provide more detailed
information.


## Running instructions

The following steps assume we are in a directory with all the relevant
programs there (`~jburguet/KOPNet/jordi_versions` for example), and
`$KEGG` points to the relevant directory (like
`KEGG=/home/huerta/_Databases/kegg.07-24/genes/fasta`).

In any machine with more than 40 GB or so:

```sh
./find_valid_ids.py $KEGG/prokaryotes.dat.gz
```

In `gpu02` to create the T5 embeddings (in an environment like what we
get with `source /home/lcano/mambaforge/bin/activate ProtTrans` and
`export
LD_LIBRARY_PATH=/home/lcano/mambaforge/envs/ProtTrans/nsight-compute/2024.1.1/host/linux-desktop-glibc_2_11_3-x64:$LD_LIBRARY_PATH`):

```sh
./t5_embedder.py $KEGG/prokaryotes.pep.gz id_ko.txt --out t5_embeddings.npz
```

In `fat01` to create the UMAP embeddings (in an environment like
`source /home/lcano/mambaforge/bin/activate python`):

```sh
./umap_embedder.py t5_embeddings.npz -n 40 -v
```

In a gpu node, train KOPNet from the UMAP embeddings:

```sh
./kopnet.py umap_embeddings.npz -e 5 -n 100 400 -l 0.005 --n-umap 30
```


## Tests

Instead of using all the proteins, we can use a subset and do the full
analysis with them. This is very useful to test the full pipeline in a
short time.

To do that, we can use the `--truncate` argument in
`find_valid_ids.py`. For example, to only use the first 5000 lines
from the `dat` file:

```sh
./find_valid_ids.py $KEGG/prokaryotes.dat.gz --truncate 5000
```

The resulting `id_ko.txt` file will have only the valid KOs (those
that appeared in at least 2 proteins) and valid protein ids (those
associated with valid KOs), taken from the first 5000 lines of the
`dat` file.

The commands for the rest of the analysis would be identical.
