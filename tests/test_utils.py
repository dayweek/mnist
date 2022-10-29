import pytest
from pathlib import Path

from data import download_file
from data import download_mnist

def test_download_file():
    download_file("https://raw.githubusercontent.com/dayweek/czech_names_vocative/master/README.md", "data")
    assert Path("data/README.md").exists(), "file not downloaded"

def test_download_mnist():
    download_mnist("data")
    assert Path("data/train-images-idx3-ubyte").exists(), "file not downloaded and decompressed"
    assert Path("data/train-labels-idx1-ubyte").exists(), "file not downloaded and decompressed"
    assert Path("data/t10k-images-idx3-ubyte").exists(), "file not downloaded and decompressed"
    assert Path("data/t10k-labels-idx1-ubyte").exists(), "file not downloaded and decompressed"
