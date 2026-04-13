"""
Class and functions to load the kegg data and serve it in a convenient
format for pytorch.
"""

import numpy as np
from torch import Tensor, FloatTensor
from torch.utils.data import Dataset, DataLoader


class KeggDataset(Dataset):
    def __init__(self, xs, ys, labels):
        self.xs = Tensor(xs)
        self.ys = ys
        self.labels = labels

    def __len__(self):
        return len(self.ys)  # number of samples

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


def get_dataloaders(emb_umap, test_size=0.25, batch_size=64, shuffle=True,
                    n_umap=30):
    """Return torch dataloaders for training and test data.

    Args:
    - n_umap: Number of umap components used.
    """
    data = np.load(emb_umap)

    xs = data['umap_embeddings']
    y_labels = data['kos']

    xs = xs[:,:n_umap]  # take only n_umap components of the umap

    labels = sorted(set(y_labels))

    label2y = {labels[i]: i for i in range(len(labels))}
    ys = np.array([label2y[label] for label in y_labels])

    unique, counts = np.unique(ys, return_counts=True)
    counts_in_order = counts[unique]  # counts for class 0, class 1, etc
    weights = 1 / counts_in_order
    weights /= weights.sum()  # normalized

    # Shuffle so we don't always train with the same data.
    if shuffle:
        xs, ys = shuffle_arrays(xs, ys)

    nclasses = len(weights)  # number of different classes
    xs_train, ys_train, xs_test, ys_test = split(xs, ys, nclasses, test_size)

    # Shuffle the training data in case the order matters (it depends).
    if shuffle:
        xs_train, ys_train = shuffle_arrays(xs_train, ys_train)

    data_train = KeggDataset(xs_train, ys_train, labels)
    data_test = KeggDataset(xs_test, ys_test, labels)

    class_weights = FloatTensor(weights)

    return (DataLoader(data_train, batch_size=batch_size),
            DataLoader(data_test, batch_size=batch_size),
            class_weights)


def shuffle_arrays(xs, ys):
    """Return arrays xs and ys with values shuffled (same way for both)."""
    indices = np.arange(len(ys))
    np.random.shuffle(indices)
    return xs[indices], ys[indices]


def split(xs, ys, nclasses=None, test_size=0.25):
    """Split the arrays xs and ys into training and test data.

    The ys must contain values 0 <= y < nclasses.

    Both parts of the split have data for all classes: (1-test_size) of
    training data (per class), and test_size of test data (per class).
    """
    nclasses = nclasses or len(set(ys))  # find nclasses if not given

    # Make partitions (lists of xs per class).
    ps = [[] for _ in range(nclasses)]
    for x, y in zip(xs, ys):
        ps[y].append(x)  # [[x0_c0, x1_c0, ...], [x0_c1, x1_c1, ...], ...]

    # Divide partitions into train data and test data.
    xs_train = [];  ys_train = []
    xs_test  = [];  ys_test  = []
    for i, p in enumerate(ps):
        n = int(len(p) * (1 - test_size))  # we take the first n for training
        xs_train.append(p[:n]);  ys_train.append([i]*n)
        xs_test.append( p[n:]);  ys_test.append( [i]*(len(p) - n))

    return (np.concat(xs_train), np.concat(ys_train),
            np.concat(xs_test),  np.concat(ys_test))
