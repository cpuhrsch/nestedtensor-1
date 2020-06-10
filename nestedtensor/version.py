__version__ = '0.0.1.dev202061021+aa5d869'
git_version = 'aa5d86911fa441ecf03a20862b28f6175c4b7adf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
