__version__ = '0.0.1.dev202081221+ce8edad'
git_version = 'ce8edad2f5056ccd54671c8c3c6b281208d36b17'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
