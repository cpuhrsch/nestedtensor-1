__version__ = '0.1.4+cf70ab7'
git_version = 'cf70ab7d20dfd62f4840cfd679d562a9c18480c6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
