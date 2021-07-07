__version__ = '0.1.4+7a4c64f'
git_version = '7a4c64f375246530a4650f409546c75de1a0206c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
