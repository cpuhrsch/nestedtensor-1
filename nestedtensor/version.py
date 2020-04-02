__version__ = '0.0.1.dev2020423+eb25890'
git_version = 'eb258900a33c34b582b93c53d8131d36b72f2c13'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
