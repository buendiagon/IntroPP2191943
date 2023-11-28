/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include "book.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define imin(a,b) (a<b?a:b)  // Define una macro para calcular el mínimo de dos valores

#define N (4*33*1024*1024)  // Tamaño total de los datos a procesar
// const int threadsPerBlock = 256;  // Número de hilos por bloque en la GPU
// const int blocksPerGrid =  // Número de bloques en la grilla de la GPU
//             imin( 32, (N/2+threadsPerBlock-1) / threadsPerBlock );

const int threadsPerBlock = 256;
const int blocksPerGrid = 32;

// Kernel de CUDA para realizar una operación de punto entre dos vectores
__global__ void dot( int size, float *a, float *b, float *c ) {
    __shared__ float cache[threadsPerBlock];  // Define una memoria caché compartida
    int tid = threadIdx.x + blockIdx.x * blockDim.x;  // Calcula el ID único para cada hilo
    int cacheIndex = threadIdx.x;  // Índice en la memoria caché para este hilo

    float temp = 0;
    while (tid < size) {  // Bucle para calcular el producto punto parcial
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;  // Salta a la siguiente posición en el vector
    }
    
    cache[cacheIndex] = temp;  // Guarda el resultado parcial en la caché

    __syncthreads();  // Sincroniza todos los hilos en el bloque

    // Reducción en paralelo para sumar todos los valores parciales
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)  // Solo un hilo escribe el resultado final para este bloque
        c[blockIdx.x] = cache[0];
}

// Estructura para almacenar los datos necesarios para cada ejecución paralela
struct DataStruct {
    int     deviceID;
    int     size;
    float   *a;
    float   *b;
    float   returnValue;
};

// Función para ejecutar el kernel dot en un hilo específico
void* routine( void *pvoidData, float *elapsedTime ) {
    cudaEvent_t start_c, end_c; // sirve para medir el tiempo de ejecución paralela (GPU)

    DataStruct  *data = (DataStruct*)pvoidData;
    HANDLE_ERROR( cudaSetDevice( data->deviceID ) );  // Establece el dispositivo GPU

    int     size = data->size;
    float   *a, *b, c, *partial_c;
    float   *dev_a, *dev_b, *dev_partial_c;

    a = data->a;
    b = data->b;
    partial_c = (float*)malloc( blocksPerGrid*sizeof(float) );  // Almacena resultados parciales

    // Asigna memoria en la GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, size*sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, size*sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_partial_c, blocksPerGrid*sizeof(float) ) );

    // Copia los datos al dispositivo GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, size*sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, size*sizeof(float), cudaMemcpyHostToDevice ) ); 

    // Sirve para medir tiempo de ejecución paralelo
    cudaEventCreate(&start_c);
    cudaEventCreate(&end_c);
    cudaEventRecord(start_c, 0);

    dot<<<blocksPerGrid,threadsPerBlock>>>( size, dev_a, dev_b, dev_partial_c );  // Ejecuta el kernel

    // Detiene el tiempo de ejecución de la GPU
    cudaEventRecord(end_c, 0);
    cudaEventSynchronize(end_c);

    float elapsedTimeLocal;
    cudaEventElapsedTime(&elapsedTimeLocal, start_c, end_c); // Calcula el tiempo de ejecución en GPU
    *elapsedTime += elapsedTimeLocal; // Suma el tiempo usado en cada GPU para obtener el total de tiempo de ejecución
    // printf("Total GPU time used in GPU %d: %f ms \n", data->deviceID, elapsedTimeLocal);

    // Copia los resultados parciales de vuelta a la CPU
    HANDLE_ERROR( cudaMemcpy( partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost ) );

    c = 0;
    for (int i=0; i<blocksPerGrid; i++) {  // Suma los resultados parciales
        c += partial_c[i];
    }
    data->returnValue = c;  // Guarda el resultado final

    // Libera la memoria en la GPU
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_partial_c ) );

    free( partial_c );  // Libera la memoria en la CPU

    return 0;
}

// Función principal
int main( void ) {    
    int n;  // Variable para saber el número de GPUs a usar
    
    printf("Enter the number of GPU to use: ");
    scanf("%d", &n);

    clock_t start, end; // sirve para medir el tiempo de ejecución total (GPU y CPU)
    start = clock();
    int deviceCount;
    HANDLE_ERROR( cudaGetDeviceCount( &deviceCount ) );  // Obtiene el número de dispositivos CUDA
    if (deviceCount < n) {
        printf( "We need at least two compute %d or greater "
                "devices, but only found %d\n", n, deviceCount );
        return 0;
    }

    float *a = (float*)malloc( sizeof(float) * N );  // Asigna memoria para el vector a
    HANDLE_NULL( a );
    float *b = (float*)malloc( sizeof(float) * N );  // Asigna memoria para el vector b
    HANDLE_NULL( b );

    for (int i=0; i<N; i++) {  // Inicializa los vectores
        a[i] = i;
        b[i] = i*2;
    }
    
    // Asigna memoria para un vector de DataStruct
    DataStruct  data[n];

    int sizePerThread = N / n; // Obtiene el tamaño de la data por GPU
    int remainder = N % n; // Obtiene el residuo de la división cuando no es exacta

    // reparte los datos uniformemente por la cantidad de GPUs solicitadas.
    for (int i = 0; i < n; ++i) {
        data[i].deviceID = i;
        data[i].size = sizePerThread;

        if(i < remainder) 
            data[i].size += 1;

        int offset = i * sizePerThread + (i < remainder ? i : remainder);
        data[i].a = a + offset;
        data[i].b = b + offset;
    }

    // Inicializa la rutina para cada una de las GPU
    float total = 0;
    float elapsedTime = 0;
    for (int i = 0; i < n; ++i) {
        routine( &(data[i]), &elapsedTime );
        total = total + data[i].returnValue;
    }

    // Libera la memoria asignada a los vectores
    free( a );
    free( b );

    // Imprime el resultado final

    // Imprime el total de tiempo empleado en correr el programa
    end = clock();
    double cpu_time_used = ((double) (end-start)) / CLOCKS_PER_SEC;
    printf("Total CPU time used: %f seconds \n", cpu_time_used);
    printf("Total GPU time used: %f ms \n", elapsedTime);
    printf( "Value calculated:  %f\n", total );
    return 0;
}
