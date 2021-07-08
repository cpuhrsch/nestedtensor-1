__version__ = '0.1.4+c9b6d27'
git_version = 'c9b6d272aec0ae1660841f03911caa6e1a3443bb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
