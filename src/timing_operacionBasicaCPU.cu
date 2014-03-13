
// Este codigo calcula (u[i]-u)/u_d_m y mide el tiempo de ejecucion

#include <stdio.h>
#include <string.h>
#include "soporte.cu"

void operacionCPU(float* u, float* lu, float u_m, float u_d, int n)
{
    int idx = 0;
    while (idx < n)
    {
     lu[idx] = (u[idx]-u_m)/u_d;
     idx += 1;
     }
}

int main(int argc, char**argv) 
{
    Timer timer;
    FILE *f;

    if (argc > 1) {
        char fname[30];
        strcpy(fname,"cpu_");
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
    
    // Declara y asigna memoria a los arreglos en el host
    
    const int u_m = 0;
    const int u_d = 255;

    int size = n*sizeof(float);

    float* h_u = (float*) malloc( size );
    for (unsigned int i=0; i < n; i++) { h_u[i] = i; }
        
    float* h_lu = (float*) malloc( size );    

    stopTime(&timer); 
    float t1 = elapsedTime(timer);
    printf("Longitud del vector: %d\n", n);
    printf("t1[s] = %f \n", t1);
    if (argc > 1) fprintf(f, "%f\n", t1);
    total_time = total_time + t1;
   
    // Calcula operacion
    
    printf("Llamando funcion operacionCPU...\n"); fflush(stdout);
    startTime(&timer);

    operacionCPU(h_u, h_lu, u_m, u_d, n);
    
    stopTime(&timer); 
    t1 = elapsedTime(timer);
    printf("t2[s] = %f \n ", t1);
    if (argc > 1) fprintf(f, "%f\n", t1);
    total_time = total_time + t1;
    
    
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
    
    printf("PRUEBA SUPERADA\n"); 
    printf("tf[s] = %f \n", total_time);
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

    if (argc > 1) fclose(f);

}