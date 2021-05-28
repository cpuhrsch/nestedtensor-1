__version__ = '0.1.4+6ae5f97'
git_version = '6ae5f973e0da84d2729c432f929cfbdb09ad2a85'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
