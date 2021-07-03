__version__ = '0.1.4+298ae34'
git_version = '298ae346dad5f08b050ec92b06444767f239f377'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
