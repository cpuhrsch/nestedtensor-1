__version__ = '0.0.1.dev2020376+8e2bdbc'
git_version = '8e2bdbc6c7add6f7d71988333561d3c430ce5bfd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
