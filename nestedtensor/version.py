__version__ = '0.0.1.dev202061022+93de36c'
git_version = '93de36c658cde4f1be304a74e5320a7ef4406e8c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
