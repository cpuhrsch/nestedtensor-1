__version__ = '0.0.1.dev2020720+5ef92bd'
git_version = '5ef92bd16f24a59e3198de6476a303c7c659a41f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
