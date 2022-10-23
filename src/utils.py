from tqdm import tqdm
from pathlib import Path
import requests
import gzip
import shutil
from PIL import Image

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

def read_labels(filepath, dir):
    with open(filepath, "rb") as f:
        # skip magic number
        bytes_read = f.read(4)
        # skip number of items
        bytes_read = f.read(4)
        
        # read labels one by one
        bytes_read = f.read(1)
        labels = {}
        file_number = 0
        while bytes_read:
            label = int.from_bytes(bytes_read, "big")
            labels[f'{dir}/{file_number}.png'] = label
            bytes_read = f.read(1)
            file_number += 1
        return labels

def create_png_files(filepath, dir):
    Path(dir).mkdir(exist_ok=True)
    
    with open(filepath, "rb") as f:
        # skip magic number
        bytes_read = f.read(4)
        # skip number of items
        bytes_read = f.read(4)
        # skip rows
        bytes_read = f.read(4)
        # skip columns
        bytes_read = f.read(4)
        
        image_size = 28 * 28
        
        # read images one by one
        bytes_read = f.read(image_size)
        file_number = 0
        while bytes_read:
            image = Image.frombuffer('L', (28, 28), bytes_read)
            image.save(f'{dir}/{file_number}.png')
            bytes_read = f.read(image_size)
            file_number += 1
    
def download_mnist(dir):
    download_file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", dir)
    download_file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", dir)
    download_file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", dir)
    download_file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", dir)
    ungz("data/train-images-idx3-ubyte.gz", "data/train-images-idx3-ubyte")
    ungz("data/train-labels-idx1-ubyte.gz", "data/train-labels-idx1-ubyte")
    ungz("data/t10k-images-idx3-ubyte.gz", "data/t10k-images-idx3-ubyte")
    ungz("data/t10k-labels-idx1-ubyte.gz", "data/t10k-labels-idx1-ubyte")
    create_png_files("data/t10k-images-idx3-ubyte", "data/test")
    create_png_files("data/train-images-idx3-ubyte", "data/train")
