__version__ = '0.0.1.dev2020870+e0deb09'
git_version = 'e0deb0948582853b9854d59bb76f33a132981abd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
