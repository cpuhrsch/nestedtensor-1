__version__ = '0.1.4+cf84e9d'
git_version = 'cf84e9dcda18320bd1e9f494f603eb0d7a3498c9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
