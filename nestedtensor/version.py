__version__ = '0.0.1.dev20205303+eefd992'
git_version = 'eefd99242a066646c376405524c46648ed6de4c4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
