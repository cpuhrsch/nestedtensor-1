__version__ = '0.1.4+75aabe1'
git_version = '75aabe1cf1071e3abc0c36e7420e7afa03addd7e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
