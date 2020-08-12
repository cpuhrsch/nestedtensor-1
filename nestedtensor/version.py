__version__ = '0.0.1.dev202081220+11ea451'
git_version = '11ea451eabb41e586903de696e9c5ee1440d528a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
