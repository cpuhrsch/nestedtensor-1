__version__ = '0.0.1.dev202010233+448e5de'
git_version = '448e5de800ae618ca43c409c48d18d5195baf955'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
