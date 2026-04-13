# Alternative versions

The following assume we are in a directory with all the relevant
programs there (`~jburguet/KOPNet/jordi_versions` for example).

In any machine with more than 40 GB or so:

```sh
conda_env
conda activate
./find_valid_ids.py  # maybe with  --truncate 1000  or similar for tests
```

In `gpu02` to create the T5 embeddings:

```sh
conda_env
source /home/lcano/mambaforge/bin/activate ProtTrans
export LD_LIBRARY_PATH=/home/lcano/mambaforge/envs/ProtTrans/nsight-compute/2024.1.1/host/linux-desktop-glibc_2_11_3-x64:$LD_LIBRARY_PATH

./t5_embedder.py --valid-ids id_ko.txt --out t5_embeddings.npz \
    /home/huerta/_Databases/kegg.07-24/genes/fasta/prokaryotes.pep.gz
```

In `fat01` to create the UMAP embeddings (and train with them KOPNet):

```sh
conda_env
source /home/lcano/mambaforge/bin/activate python

./umap_embedder.py t5_embeddings.npz -n 40 -v

./kos_nn_w_np.py umap_embeddings.npz \
    -e 3 -n 100 400 -l 0.005 \
    -o trained_model
```
