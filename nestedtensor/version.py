__version__ = '0.0.1.dev202081420+e5e5422'
git_version = 'e5e5422399887e0efc49dddff49627c0a3f55e05'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
