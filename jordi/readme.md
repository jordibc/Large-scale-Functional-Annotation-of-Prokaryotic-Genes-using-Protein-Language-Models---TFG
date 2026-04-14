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


## Running instructions

The following assume we are in a directory with all the relevant
programs there (`~jburguet/KOPNet/jordi_versions` for example).

In any machine with more than 40 GB or so:

```sh
conda_env
conda activate
./find_valid_ids.py  # use  --truncate 1000  for tests with the first 1000
```

In `gpu02` to create the T5 embeddings:

```sh
conda_env
source /home/lcano/mambaforge/bin/activate ProtTrans
export LD_LIBRARY_PATH=/home/lcano/mambaforge/envs/ProtTrans/nsight-compute/2024.1.1/host/linux-desktop-glibc_2_11_3-x64:$LD_LIBRARY_PATH

./t5_embedder.py --valid-ids id_ko.txt --out t5_embeddings.npz \
    /home/huerta/_Databases/kegg.07-24/genes/fasta/prokaryotes.pep.gz
```

In `fat01` to create the UMAP embeddings:

```sh
conda_env
source /home/lcano/mambaforge/bin/activate python

./umap_embedder.py -v -n 40 t5_embeddings.npz
```

In a gpu node, train KOPNet from the UMAP embeddings:

```sh
./kopnet.py -e 5 -n 100 400 -l 0.005 --n-umap 30 umap_embeddings.npz
```
