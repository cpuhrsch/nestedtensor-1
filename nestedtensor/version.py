__version__ = '0.0.1.dev202061018+8da91d1'
git_version = '8da91d1ae3331f72f6132cb647020e5067818e62'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
