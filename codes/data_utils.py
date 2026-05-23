"""MNIST loading, fixed train/val split, optional download."""
import gzip
import os
import pickle
import urllib.request
from struct import unpack

import numpy as np

MNIST_DIR = os.path.join(os.path.dirname(__file__), "dataset", "MNIST")
IDX_PATH = os.path.join(os.path.dirname(__file__), "idx.pickle")
SEED = 309

URLS = {
    "train-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_mnist():
    os.makedirs(MNIST_DIR, exist_ok=True)
    for name, url in URLS.items():
        path = os.path.join(MNIST_DIR, name)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved {path}")


def _read_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = unpack(">4I", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = unpack(">2I", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist_splits(force_download=False):
    if force_download:
        download_mnist()
    train_img_path = os.path.join(MNIST_DIR, "train-images-idx3-ubyte.gz")
    if not os.path.isfile(train_img_path):
        download_mnist()

    train_imgs = _read_images(train_img_path)
    train_labs = _read_labels(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte.gz"))
    test_imgs = _read_images(os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte.gz"))
    test_labs = _read_labels(os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte.gz"))

    if os.path.isfile(IDX_PATH):
        with open(IDX_PATH, "rb") as f:
            idx = pickle.load(f)
    else:
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(np.arange(train_imgs.shape[0]))
        with open(IDX_PATH, "wb") as f:
            pickle.dump(idx, f)

    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    scale = 255.0
    train_imgs = train_imgs.astype(np.float64) / scale
    valid_imgs = valid_imgs.astype(np.float64) / scale
    test_imgs = test_imgs.astype(np.float64) / scale

    return (train_imgs, train_labs), (valid_imgs, valid_labs), (test_imgs, test_labs)
