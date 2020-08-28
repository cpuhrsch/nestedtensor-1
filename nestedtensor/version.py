__version__ = '0.0.1.dev202082722+f994095'
git_version = 'f994095e3495f77f0e568c0648e4266532d79fdb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
