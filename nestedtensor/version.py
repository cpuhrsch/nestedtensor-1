__version__ = '0.1.4+716aa03'
git_version = '716aa03edcecdd323fbc556ff469078510a5ac2e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
