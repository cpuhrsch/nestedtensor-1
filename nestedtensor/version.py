__version__ = '0.1.4+4836928'
git_version = '4836928a4ea3ef876e7a69ab229ed3bde4bd3eec'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
