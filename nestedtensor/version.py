__version__ = '0.1.4+8dcbff1'
git_version = '8dcbff1f9387341f0cbb1a5ba5db3d5caba514f1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
