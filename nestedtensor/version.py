__version__ = '0.0.1.dev202061020+95ca5bc'
git_version = '95ca5bc85ec2d3662ddbb3b05195b697b3174335'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
