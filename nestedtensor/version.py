__version__ = '0.0.1.dev20205303+f900104'
git_version = 'f900104027f0355b6e7cc51e23a4e31dd0b17429'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
