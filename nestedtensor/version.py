__version__ = '0.1.4+123ec12'
git_version = '123ec128a2dc98c7550236c26d80e303f5e5c1c6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
