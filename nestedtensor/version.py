__version__ = '0.1.4+d3d0d3f'
git_version = 'd3d0d3f25ed1f38def004abba0ada41276f42307'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
