
#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

n = 25
h_u = arange(n).astype(np.float32)
u_m = np.float32(0)
u_d = np.float32(255)

d_u = cuda.mem_alloc(h_u.nbytes)

cuda.memcpy_htod(d_u, h_u)

mod = SourceModule("""
__global__ void operacionKernelGPU(float* u, float u_m, float u_d, int n)
{
int idx = threadIdx.x + blockDim.x * blockIdx.x;
if (idx < n)
    u[idx] = (u[idx]-u_m)/u_d;
}
""")
operacionKernelGPU = mod.get_function("operacionKernelGPU")
operacionKernelGPU(d_u, u_m, u_d, np.uint32(h_u.size), block=(32, 1, 1))

u = numpy.empty_like(h_u)
cuda.memcpy_dtoh(u, d_u)
print u
print (h_u-u_m)/u_d
!codes/timing_operacionBasicaGPU