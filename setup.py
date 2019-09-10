"""
VGGish setup
============
"""
import urllib.request

from setuptools import setup

urllib.request.urlretrieve(
    "https://storage.googleapis.com/audioset/vggish_model.ckpt", "src/vggish/model.ckpt"
)

setup()
