__version__ = '0.1.4+3d6f444'
git_version = '3d6f444c772d32acf986232680393c9c318c6c5b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
