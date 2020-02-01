__version__ = '0.0.1.dev20202120+63f281f'
git_version = '63f281f2832a2629f689bec7bd77b7ab1a9c686c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
