__version__ = '0.1.4+92e1b70'
git_version = '92e1b7058c01ec17c43fa0ce1384cd4ae4487f06'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
