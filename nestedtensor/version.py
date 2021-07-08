__version__ = '0.1.4+b53c427'
git_version = 'b53c427c40e1b9e6b4a5117e1d7354fd0e3b7940'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
