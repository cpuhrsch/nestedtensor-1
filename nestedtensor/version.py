__version__ = '0.0.1.dev202081022+49e9561'
git_version = '49e956174b3e3a84f937777cda78726f448b6592'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
