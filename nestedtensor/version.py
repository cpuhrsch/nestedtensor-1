__version__ = '0.1.4+2587f1a'
git_version = '2587f1a8bb64bf0f787bbd6ef91856a4e40de8b6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
