from tqdm import tqdm
import requests

def download_mnist_data():
    URL = "https://s3.amazonaws.com/fast-ai-sample/mnist_sample.tgz"

    r = requests.get(URL, stream=True)
    total = int(r.headers.get('content-length', 0))

    with open('data/mnist_sample.tgz', 'wb') as f, tqdm(total=total, unit="B", unit_divisor=1024) as t:
        for chunk in r.iter_content(chunk_size=1024):
            t.update(1024)
            f.write(chunk)