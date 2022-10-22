from tqdm import tqdm
from pathlib import Path
import requests

import gzip
import shutil

def download_file(url, dir):
    dest_file = Path(dir) / Path(url).name
    r = requests.get(url, stream=True)
    total = int(r.headers.get('content-length', 0))

    with open(dest_file, 'wb') as f, tqdm(total=total, unit="B", unit_divisor=1024) as t:
        for chunk in r.iter_content(chunk_size=1024):
            t.update(1024)
            f.write(chunk)

# https://stackoverflow.com/a/44712152
def ungz(filepath_in, filepath_out):
    with gzip.open(filepath_in, 'rb') as f_in:
        with open(filepath_out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def download_mnist(dir):
    download_file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", dir)
    download_file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", dir)
    download_file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", dir)
    download_file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", dir)
    ungz("data/train-images-idx3-ubyte.gz", "data/train-images-idx3-ubyte")
    ungz("data/train-labels-idx1-ubyte.gz", "data/train-labels-idx1-ubyte")
    ungz("data/t10k-images-idx3-ubyte.gz", "data/t10k-images-idx3-ubyte")
    ungz("data/t10k-labels-idx1-ubyte.gz", "data/t10k-labels-idx1-ubyte")