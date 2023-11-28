#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // Get the number of CUDA-capable devices

    if (deviceCount == 0) {
        printf("No CUDA-capable devices were detected\n");
        return 1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i); // Get properties of the ith device

        printf("Device %d: %s\n", i, deviceProp.name);
        printf("  Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);

        // Assuming a 3D grid for blocks, print the maximum dimensions
        printf("  Maximum dimensions of block: x = %d, y = %d, z = %d\n",
            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

        // Assuming a 3D grid for grid size, print the maximum dimensions
        printf("  Maximum dimensions of grid: x = %d, y = %d, z = %d\n",
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }

    return 0;
}
