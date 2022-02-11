__version__ = '0.1.4+c5434a9'
git_version = 'c5434a962c33ecfbc52636e9d68fb7958e6791bb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
