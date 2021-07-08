__version__ = '0.1.4+2435394'
git_version = '2435394d69745ca671fc94d7c84ccf48c3417d01'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
