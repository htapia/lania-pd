
// Este codigo calcula (u[i]-u)/u_d_m en el dispositivo GPU
// y mide el tiempo de ejecucion

#include <stdio.h>
#include <string.h>
#include "soporte.cu"

__global__ void operacionKernelGPU(float* u, float* lu, float u_m, float u_d, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) lu[idx] = (u[idx]-u_m)/u_d;
}

int main(int argc, char**argv) 
{
    Timer timer;

    FILE *f;

    if (argc > 1) {
        char fname[30];
        strcpy(fname,"gpu_");
        strcat(fname,argv[1]);
        f = fopen(fname, "w");
        if (f == NULL)
            {
                printf("Error abriendo archivo\n");
                exit(1);
            }
        }
    float total_time = 0.0;
    
    printf("\nInicializando...\n"); fflush(stdout);
    startTime(&timer);
 
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
    
// Asignando memoria a las variables u_h, lu_h ----------------------------------------------------------
    
    const int u_m = 0;
    const int u_d = 255;

    int size = n*sizeof(float);

    float* h_u = (float*) malloc( size );
    for (unsigned int i=0; i < n; i++) { h_u[i] = i; }
        
    float* h_lu = (float*) malloc( size );

    stopTime(&timer); 
    float t1 = elapsedTime(timer);
    printf("Longitud del vector: %d\n", n);
    printf("t1 = %f s\n", t1);
    if (argc > 1) fprintf(f, "%f\n", t1);
    total_time = total_time + t1;
    
//  Declaracion de vectores/arreglos a usarse en el GPU ----------------------------------------------------------

    printf("Asignando memoria en el GPU...\n"); fflush(stdout);
    startTime(&timer);
    
    float* d_u;
    cudaMalloc((void**)&d_u, size);
    float* d_lu;
    cudaMalloc((void**)&d_lu, size);
    // cudaDeviceSynchronize();

    stopTime(&timer); 
    t1 = elapsedTime(timer);
    printf("t2 = %f s\n", t1);
    if (argc > 1) fprintf(f, "%f\n", t1);
    total_time = total_time + t1;
    
//  Copiando data del host al GPU ----------------------------------------------------------

    printf("Copiando datos desde el host al GPU...\n"); fflush(stdout);
    startTime(&timer);
    
    cudaMemcpy( d_u, h_u, size, cudaMemcpyHostToDevice );
    
    stopTime(&timer); 
    t1 = elapsedTime(timer);
    printf("t3 = %f s\n", t1);
    if (argc > 1) fprintf(f, "%f\n", t1);
    total_time = total_time + t1;
    
    // Launch kernel ----------------------------------------------------------

    printf("Lanzando kernel...\n"); fflush(stdout);
    startTime(&timer);
    
    operacionKernelGPU<<<ceil(n/1024.0), 1024>>>(d_u, d_lu, u_m, u_d, n);
    
    stopTime(&timer); 
    t1 = elapsedTime(timer);
    printf("t4 = %f s\n", t1);
    if (argc > 1) fprintf(f, "%f\n", t1);
    total_time = total_time + t1;
    
//  Copiando variables del GPU al host ----------------------------------------------------------

    printf("Copiando datos desde el GPU al host...\n"); fflush(stdout);
    startTime(&timer);
    
    cudaMemcpy( h_lu, d_lu, size, cudaMemcpyDeviceToHost );
    
    stopTime(&timer); 
    t1 = elapsedTime(timer);
    printf("t5 = %f s\n", t1);
    if (argc > 1) fprintf(f, "%f\n", t1);
    total_time = total_time + t1;
        
// verificando resultados ----------------------------------------------------------

    const float toleranciaRelativa = 1e-4;
    
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
    
    printf("PRUEBA SUPERADA\n");
    printf("tf = %f s\n", total_time);
    if (argc > 1) fprintf(f, "%f\n", total_time);

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

    if (argc > 1) fclose(f);
    
}