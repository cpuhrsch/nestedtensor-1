__version__ = '0.0.1.dev202072321+fa32d73'
git_version = 'fa32d731db91002b83913ad9b2d2d925b651ad68'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
