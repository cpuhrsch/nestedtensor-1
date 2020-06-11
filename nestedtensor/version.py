__version__ = '0.0.1.dev202061123+07ccb66'
git_version = '07ccb66a6569e0178363e18b287f472ea23783d4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
