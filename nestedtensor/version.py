__version__ = '0.1.4+2c369fd'
git_version = '2c369fd8107a47133893e39fbaf53d0663bde50d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
