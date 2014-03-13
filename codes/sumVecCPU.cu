// # %%file codes/sumVecCPU.cu
#include <stdio.h>
#include <string.h>
#include "soporte.cu"

// Función que calcula la suma C_h = A_h + B_h en el CPU
// compilar usando nvcc codes/sumVecCPU.cu -o codes/sumVecCPU

void vecSumCPU(float* A, float* B, float* C, int n)
{
    // for (int i=0; i < n; i++) C[i] = A[i] + B[i];
    int tid = 0;    // el primer CPU del sistema
    while (tid < n) {
        C[tid] = A[tid] + B[tid];
        tid += 1;   // incrementamos en uno pues solo estamos usando un CPU
    }
    
}

int main(int argc, char**argv) 
{
    Timer timer;
    FILE *f;

    if (argc > 1) {
        char fname[30];
        strcpy(fname,"sumcpu_");
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
        n = 10000;
    } else if(argc == 2) {
        n = atoi(argv[1]);
    } else {
        printf("\n    Parametros no validos!"
           "\n    Uso: ./vecSumCPU               # Vector of longitud 10,000"
           "\n    Uso: ./vecSumCPU <m>           # Vector of longitud m"
           "\n");
        exit(0);
    }

    int size = n*sizeof(float);

    float* A_h = (float*) malloc( size );
    for (unsigned int i=0; i < n; i++) { A_h[i] = (rand()%100)/100.00; }

    float* B_h = (float*) malloc( size );
    for (unsigned int i=0; i < n; i++) { B_h[i] = (rand()%100)/100.00; }

    float* C_h = (float*) malloc( size );

    stopTime(&timer); 
    float t1 = elapsedTime(timer);
    printf("Longitud del vector: %d\n", n);
    printf("t1[s] = %f \n", t1);
    if (argc > 1) fprintf(f, "%f\n", t1);
    total_time = total_time + t1;
    
    printf("Llamando funcion operacionCPU...\n"); fflush(stdout);
    startTime(&timer);

    vecSumCPU(A_h, B_h, C_h, n);
    
    stopTime(&timer); 
    t1 = elapsedTime(timer);
    printf("t2[s] = %f \n ", t1);
    if (argc > 1) fprintf(f, "%f\n", t1);
    total_time = total_time + t1;

    const float toleranciaRelativa = 1e-6;

   for(int i=0; i < n; i++) {
     float sum = A_h[i] + B_h[i];
     float relativeError = (sum - C_h[i])/sum;
     if (relativeError > toleranciaRelativa
       || relativeError < -toleranciaRelativa) {
       printf("PRUEBA FALLIDA\n\n");
       exit(0);
     }
   }
   printf("PRUEBA SUPERADA\n\n");
   printf("tf[s] = %f \n", total_time);
   if (argc > 1) fprintf(f, "%f\n", total_time); 

    free(A_h);
    free(B_h);
    free(C_h);

  if (argc > 1) fclose(f);
}