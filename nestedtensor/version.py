__version__ = '0.0.1.dev202061021+565b2bc'
git_version = '565b2bc5c01d3fafb3353e23c0ccd5f79e92cd0f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
