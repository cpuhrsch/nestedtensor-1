__version__ = '0.1.4+5e2a8a1'
git_version = '5e2a8a171daecf9a484dd5a6d747bb19330e693b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
