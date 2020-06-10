__version__ = '0.0.1.dev202061020+d10bded'
git_version = 'd10bdedf2cdfb8de566aaf1adf214bd0507f28c8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
