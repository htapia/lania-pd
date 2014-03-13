// Este codigo obtiene la derivada de la funcion u(x) = x*x
// utilizando diferencias finitas, donde el error es proporcional
// a dx

#include <stdio.h>


void derivCPU(float* u_h, float* du_h, float dx, int n)
{
    // notar que este loop empieza en 1, porque?
    for (int i=1; i < n; i++) du_h[i] = (u_h[i] - u_h[i-1])/dx;
}

int main(int argc, char**argv) 
{
    unsigned int n;
    if(argc == 1) {
        n = 41;
    } else if(argc == 2) {
        n = atoi(argv[1]);
    } else {
        printf("\n    Parametros no validos!"
           "\n    Uso: ./derivCPU               # Vector of longitud 10,000"
           "\n    Uso: ./derivCPU <m>           # Vector of longitud m"
           "\n");
        exit(0);
    }

    float L = 1.;
    float dx = L/(n-1);
    
// x_h, u_h, du_h

    int size = n*sizeof(float);

// Particion del intervalo

    float* x_h = (float*) malloc( size );
    for (unsigned int i=0; i < n; i++) { x_h[i] = i*dx; }
        
    float* u_h = (float*) malloc( size );
    for (unsigned int i=0; i < n; i++) { u_h[i] = x_h[i]*x_h[i]; }

    float* du_h = (float*) malloc( size );
    for (unsigned int i=0; i < n; i++) { du_h[i] = u_h[i]; }

    derivCPU(u_h, du_h, dx, n);

  //  for(int i=0; i < n; i++) {
  //   float diff = dx - (2*x_h[i] - du_h[i]);
  //   printf("%f %f %f %f\n", du_h[i], 2*x_h[i], diff, diff/dx);
  // }

  const float toleranciaRelativa = 1e-4;

   for(int i=1; i < n; i++) {
     float deriv = 2*x_h[i];
     float relativeError = dx - (deriv - du_h[i]);
     if (relativeError > toleranciaRelativa
       || relativeError < -toleranciaRelativa) {
       printf("PRUEBA FALLIDA\n\n");
       exit(0);
     }
   }
   printf("PRUEBA SUPERADA\n\n");    

    free(u_h);
    free(du_h);
    free(x_h);
}