__version__ = '0.1.4+11d7635'
git_version = '11d7635df4d133271b2d56d46dbb57b526fd2a2d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
