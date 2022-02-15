__version__ = '0.1.4+a72fbc9'
git_version = 'a72fbc92414a2fefb8a821f3783ca61af53b8f35'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
