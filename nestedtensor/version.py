__version__ = '0.0.1.dev20205306+188daf3'
git_version = '188daf3eb2d1d9a7dedecce9e37cf77713e68b34'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
