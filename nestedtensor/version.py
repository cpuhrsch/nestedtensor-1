__version__ = '0.0.1.dev20205303+e6058a1'
git_version = 'e6058a17ea5b781ee0dcd5f7e8f1982e0ee34c27'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
