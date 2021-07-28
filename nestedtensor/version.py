__version__ = '0.1.4+ac24928'
git_version = 'ac24928271e0bb8bb5b77f8dc4ee9d8b1307c150'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
