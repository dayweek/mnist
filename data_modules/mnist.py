from tqdm.autonotebook import tqdm
from pathlib import Path
import requests
import gzip
import shutil
from PIL import Image
import torch
from torch import tensor
from torchvision import transforms
convert_tensor = transforms.ToTensor()

from torch.utils.data import DataLoader

class MNISTDataModule():
    def __init__(self, dir):
        self.dir = dir

    def prepare_data(self):
        if not Path("data/train-labels-idx1-ubyte").exists():
            Path(dir).mkdir(exist_ok=True)
            download_file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", dir)
            download_file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", dir)
            download_file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", dir)
            download_file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", dir)
            ungz(f"{self.dir}/train-images-idx3-ubyte.gz", f"{self.dir}/train-images-idx3-ubyte")
            ungz(f"{self.dir}/train-labels-idx1-ubyte.gz", f"{self.dir}/train-labels-idx1-ubyte")
            ungz(f"{self.dir}/t10k-images-idx3-ubyte.gz", f"{self.dir}/t10k-images-idx3-ubyte")
            ungz(f"{self.dir}/t10k-labels-idx1-ubyte.gz", f"{self.dir}/t10k-labels-idx1-ubyte")
            create_png_files(f"{self.dir}/t10k-images-idx3-ubyte", f"{self.dir}/test")
            create_png_files(f"{self.dir}/train-images-idx3-ubyte", f"{self.dir}/train")
    
    def setup(self):
        train_labels = read_labels('data/train-labels-idx1-ubyte', 'train')
        keys = list(train_labels)

        train_x = torch.stack([convert_tensor(Image.open(Path("data") / key)) for key in keys])
        train_y = tensor([train_labels[key] for key in keys])

        self.train_ds = BaseDataset(train_x, train_y)

        test_labels = read_labels('data/t10k-labels-idx1-ubyte', 'test')
        test_keys = list(test_labels)

        test_x = torch.stack([convert_tensor(Image.open(Path("data") / key)) for key in test_keys])
        test_y = tensor([test_labels[key] for key in test_keys])

        self.val_ds = BaseDataset(test_x, test_y)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=40, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.train_ds, batch_size=40, shuffle=True)



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
    

    print("MNIST data downloaded and processed.")

def load_test_dataset():
    if not Path("data/t10k-labels-idx1-ubyte").exists():
        download_mnist("data")

    test_labels = read_labels('data/t10k-labels-idx1-ubyte', 'test')
    test_keys = list(test_labels)

    test_x = torch.stack([convert_tensor(Image.open(Path("data") / key)) for key in test_keys])
    test_y = tensor([test_labels[key] for key in test_keys])

    return BaseDataset(test_x, test_y)

def load_train_dataset():
    if not Path("data/train-labels-idx1-ubyte").exists():
        download_mnist("data")

    train_labels = read_labels('data/train-labels-idx1-ubyte', 'train')
    keys = list(train_labels)

    train_x = torch.stack([convert_tensor(Image.open(Path("data") / key)) for key in keys])
    train_y = tensor([train_labels[key] for key in keys])

    return BaseDataset(train_x, train_y)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

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