__version__ = '0.1.4+94f1258'
git_version = '94f1258e347dd94fe440102460352f313b633062'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
