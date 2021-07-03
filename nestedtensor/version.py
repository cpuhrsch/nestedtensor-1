__version__ = '0.1.4+baf7a49'
git_version = 'baf7a49c8d4f1e4302d3bd978f199e02dcf4b6da'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
