
// Este codigo calcula (u[i]-u)/u_d_m en el dispositivo GPU

#include <stdio.h>


__global__ void operacionKernelGPU(float* u, float* lu, float u_m, float u_d, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) lu[idx] = (u[idx]-u_m)/u_d;
}

int main(int argc, char**argv) 
{
    unsigned int n;
    if(argc == 1) {
        n = 25;
    } else if(argc == 2) {
        n = atoi(argv[1]);
    } else {
        printf("\n    Parametros no validos!"
           "\n    Uso: ./derivCPU               # Vector of longitud 10,000"
           "\n    Uso: ./derivCPU <m>           # Vector of longitud m"
           "\n");
        exit(0);
    }
    
// u_h, lu_h
    
    const int u_m = 0;
    const int u_d = 255;

    int size = n*sizeof(float);

    float* h_u = (float*) malloc( size );
    for (unsigned int i=0; i < n; i++) { h_u[i] = i; }
        
    float* h_lu = (float*) malloc( size );

//  declaracion de vectores/arreglos a usarse en el GPU

    float* d_u;
    cudaMalloc((void**)&d_u, size);
    float* d_lu;
    cudaMalloc((void**)&d_lu, size);
    // cudaDeviceSynchronize();
    cudaMemcpy( d_u, h_u, size, cudaMemcpyHostToDevice );
    
    
    operacionKernelGPU<<<ceil(n/256.0),256>>>(d_u, d_lu, u_m, u_d, n);
    
    cudaMemcpy( h_lu, d_lu, size, cudaMemcpyDeviceToHost );
        
    const float toleranciaRelativa = 1e-4;
    
    // verificar
    for(int i=0; i<n; i++)
    {
     float operBasic = (h_u[i]-u_m)/u_d;
     float errorRelativo = (operBasic-h_lu[i])/operBasic;
     if (errorRelativo > toleranciaRelativa
       || errorRelativo < -toleranciaRelativa) {
       printf("PRUEBA FALLIDA\n\n");
       exit(0);
       }
     }
    
    printf("PRUEBA SUPERADA\n\n");    

    if (n==25) 
    {
     for(int i=0; i<n; i++)
     {
      printf("%10.8f\t",h_lu[i]);
     }
     printf("\n");
     }
    
    free(h_u);
    free(h_lu);
    
    cudaFree(d_u);
    cudaFree(d_lu);
}