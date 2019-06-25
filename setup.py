import os
import urllib.request
from setuptools import setup

root = os.path.dirname(os.path.realpath(__file__))


def download_checkpoint():
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/audioset/vggish_model.ckpt",
        "audioset/vggish_model.ckpt",
    )


download_checkpoint()

setup()
