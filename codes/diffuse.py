
import numpy as np
from numba import double, jit, autojit


# python
def diffuse_python(u):
    
    nx,ny = u.shape
    alpha = 0.645
    dx = 3.5/(nx-1)
    dy = 3.5/(ny-1)
#     sigma = 0.25
#     dt = sigma*dx*dx/alpha
    dt = np.float64(1e-05)
    time = np.float64(0.4)
    nt = np.int64(np.ceil(time/dt))
#     print nt

#     u = np.zeros((ny,nx))
    for n in range(nt+1): 
        un = u.copy()
        u[1:-1,1:-1]=un[1:-1,1:-1]+alpha*dt/dx**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1])+alpha*dt/dy**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2])
        u[0,:]=200
        u[-1,:]=0
        u[:,0]=200
        u[:,-1]=0
    
    return u

fast_diffuse = jit('f8(f8[:,:])')(diffuse_python)
flex_diffuse = autojit(diffuse_python)

# u_cpu = diffuse_python(64,64)
# plot_temp(u_cpu)


# pycuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod1 = SourceModule("""
__global__ void copy_array (float *u, float *u_prev, int N, int BSZ)
{	
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((x<N) && (y<N)) {
		int idx = x + y * blockDim.x*gridDim.x;
		u_prev[idx] = u[idx];
	}
}
""")

mod2 = SourceModule("""
__global__ void update (float *u, float *u_prev, int N, float h, float dt, float alpha, int BSZ)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if (x>0 && (x<N-1) && y>0 && (y<N-1)) {
		int idx = x + y * blockDim.x*gridDim.x;
		u[idx] = u_prev[idx] + alpha*dt/(h*h) * (u_prev[idx+1] + u_prev[idx-1] + u_prev[idx+N] + u_prev[idx-N] - 4*u_prev[idx]);
	}
}
""")

copy_array = mod1.get_function("copy_array")
update = mod2.get_function('update')

def diffuse_pycuda(u):
    
    nx,ny = np.int32(u.shape)
    alpha = np.float32(0.645)
    dx = np.float32(3.5/(nx-1))
    dy = np.float32(3.5/(ny-1))
    dt = np.float32(1e-05)
    time = np.float32(0.4)
    nt = np.int32(np.ceil(time/dt))
#     print nt
    
    u[0,:]=200
    u[:,0]=200  
    
    u = u.astype(np.float32)
    
    u_prev = u.copy()    
    
    u_d = cuda.mem_alloc(u.size*u.dtype.itemsize)
    u_prev_d = cuda.mem_alloc(u_prev.size*u_prev.dtype.itemsize)
    cuda.memcpy_htod(u_d, u)
    cuda.memcpy_htod(u_prev_d, u_prev)

    BLOCKSIZE = 16
    gridSize = (int(np.ceil(nx/BLOCKSIZE)),int(np.ceil(nx/BLOCKSIZE)),1)
    blockSize = (BLOCKSIZE,BLOCKSIZE,1)

    for t in range(nt+1):
        copy_array(u_d, u_prev_d, nx, np.int32(BLOCKSIZE), block=blockSize, grid=gridSize)
        update(u_d, u_prev_d, nx, dx, dt, alpha, np.int32(BLOCKSIZE), block=blockSize, grid=gridSize)
    
    cuda.memcpy_dtoh(u, u_d)
    
    return u

super_fast_diffuse = jit('f8(f8[:,:])')(diffuse_pycuda)
super_flex_diffuse = autojit(diffuse_pycuda)

# u_gpu = diffuse_pycuda(64,64)
# plot_temp(u_gpu)

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm

def plot_temp(u):
    nx,ny = u.shape
    x = np.linspace(0,3.5,nx)
    y = np.linspace(0,3.5,ny)
    X,Y = np.meshgrid(x,y)
    #         surf = 
    plt.pcolormesh(X,Y,u[:], cmap=cm.gnuplot)
    plt.axes().set_aspect('equal', 'box')
    plt.show()
#         plt.savefig('sol_%s.png'%nt)