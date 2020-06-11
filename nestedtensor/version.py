__version__ = '0.0.1.dev202061123+db1ceae'
git_version = 'db1ceae6411d3348763d65f9e1177a21178a6610'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
