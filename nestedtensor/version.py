__version__ = '0.1.4+7bedf3d'
git_version = '7bedf3d54392ca592639e6c6307e735551646cb4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
