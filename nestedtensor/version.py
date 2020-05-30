__version__ = '0.0.1.dev20205304+e10f37d'
git_version = 'e10f37de8775cd981ec42ee1e099ecbbdc2e2904'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
