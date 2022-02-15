__version__ = '0.1.4+5ea8cd5'
git_version = '5ea8cd503f8d7a5d8b8913fc04c8499b3b540d3d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
