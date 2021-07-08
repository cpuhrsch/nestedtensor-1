__version__ = '0.1.4+3249944'
git_version = '324994412774767c49b5fbbbc68978b9e821f0f3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
