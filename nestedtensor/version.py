__version__ = '0.0.1.dev20208273+370e374'
git_version = '370e3749967c71c143c6ff3e482b330c67138b7b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
