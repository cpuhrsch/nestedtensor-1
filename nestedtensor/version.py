__version__ = '0.1.4+092163c'
git_version = '092163cc6bf7dfaa72ab840db203905309685ce8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
