//este codigo sirve para realizar la suma de dos arreflos en paralelo haciedo uso de GPU

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define NB 32  // Número de bloques por rejilla
#define NT 500 // Número de hilos por bloque
#define N NB * NT // Tamaño total del arreglo

__global__ void add(double *a, double *b, double *c);

// Función del kernel que realiza la suma en paralelo
__global__ void add(double *a, double *b, double *c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(void)
{
    // Declaración de arreglos en el host y en el dispositivo
    double *a, *b, *c;
    double *dev_a, *dev_b, *dev_c;

    // Asignación de memoria en el host
    a = (double *)malloc(N * sizeof(double));
    b = (double *)malloc(N * sizeof(double));
    c = (double *)malloc(N * sizeof(double));

    // Asignación de memoria en el dispositivo (GPU)
    cudaMalloc((void **)&dev_a, N * sizeof(double));
    cudaMalloc((void **)&dev_b, N * sizeof(double));
    cudaMalloc((void **)&dev_c, N * sizeof(double));


    // Inicialización de los arreglos 'a' y 'b' en el host
    for (int i = 0; i < N; i++)
    {
        a[i] = (double)i;
        b[i] = (double)i * 2;
    }

    // Copia de los arreglos 'a' y 'b' desde el host al dispositivo
    cudaMemcpy(dev_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(double), cudaMemcpyHostToDevice);

 
   

    // Sirve para medir tiempo de ejecución paralelo
    cudaEvent_t start_c, end_c;
    cudaEventCreate(&start_c);
    cudaEventCreate(&end_c);
    cudaEventRecord(start_c, 0);

    // Llamada al kernel 'add' con NB bloques y NT hilos por bloque
    for (int i = 0; i < 10000; ++i)
        add<<<NB, NT>>>(dev_a, dev_b, dev_c);

    
     // Detiene el tiempo de ejecución de la GPU
    cudaEventRecord(end_c, 0);
    cudaEventSynchronize(end_c);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start_c, end_c);

    // Copia del resultado desde el dispositivo al host
    cudaMemcpy(c, dev_c, N * sizeof(double), cudaMemcpyDeviceToHost);

  	// verificar kis resultados
    for (int i=0; i<N; i++) {
         printf( "%g + %g = %g\n", a[i], b[i], c[i] );
     }

    printf("GPU done\n");
    printf("Total GPU time used: %f ms \n", elapsedTime);
    	
    // Liberación de memoria en el dispositivo
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // Liberación de memoria en el host
    free(a);
    free(b);
    free(c);

    return 0;
}

