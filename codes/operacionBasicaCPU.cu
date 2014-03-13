
// Este codigo calcula (u[i]-u)/u_d_m

#include <stdio.h>


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

    float* u_h = (float*) malloc( size );
    for (unsigned int i=0; i < n; i++) { u_h[i] = i; }
        
    float* lu_h = (float*) malloc( size );

    operacionCPU(u_h, lu_h, u_m, u_d, n);
    
    const float toleranciaRelativa = 1e-4;
    
    // verificar
    for(int i=0; i<n; i++)
    {
     float operBasic = (u_h[i]-u_m)/u_d;
     float errorRelativo = (operBasic-lu_h[i])/operBasic;
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
      printf("%10.8f\t",lu_h[i]);
     }
     printf("\n");
     }
    
    free(u_h);
    free(lu_h);
}