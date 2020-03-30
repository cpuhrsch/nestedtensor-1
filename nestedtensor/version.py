__version__ = '0.0.1.dev20203301+0e68070'
git_version = '0e6807062fcfee510ca8417bec47aa901792aa9e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
