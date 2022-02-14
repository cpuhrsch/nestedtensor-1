__version__ = '0.1.4+2c54429'
git_version = '2c544293914b639d019f7763c4c0e6379e32b780'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
