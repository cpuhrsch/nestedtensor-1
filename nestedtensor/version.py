__version__ = '0.1.4+a386308'
git_version = 'a386308c2c226cd63dce15d15ae7e6280904a4a8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
