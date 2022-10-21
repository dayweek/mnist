import pytest
from os import path
from download_mnist_data import download_mnist_data

def test_download_mnist_data():
    download_mnist_data()
    assert path.exists("data/mnist_sample.tgz"), "mnist data not downloaded"
