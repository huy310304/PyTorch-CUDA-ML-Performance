#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <time.h>

// replace with larger vector size to test
#define N (1024) 

// add function cuda
__global__ void add(float *a, float *b, float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the thread's unique global index
    c[tid] = a[tid] + b[tid];
}

int main(int argc, char *argv[])
{
    cudaSetDevice(0);

    int i;
    float *a, *b, *c;
    float *dev_a, *dev_b, *dev_c;

    // allocate memory on the CPU
    a = (float *) malloc(N*sizeof(float));
    b = (float *) malloc(N*sizeof(float));
    c = (float *) malloc(N*sizeof(float));

    // allocate the memory on the GPU
    cudaMalloc((void **) &dev_a, N * sizeof(float));
    cudaMalloc((void **) &dev_b, N * sizeof(float));
    cudaMalloc((void **) &dev_c, N * sizeof(float));

    // fill the arrays 'a' and 'b' on the CPU
    for (i=0; i<N; i++) {
        a[i] = sin(i) * sin(i);
        b[i] = cos(i) * cos(i);
    }

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    clock_t startTime = clock();

    // add the arrays 'a' and 'b' on the GPU
    add<<<N / 1024,1024>>>(dev_a, dev_b, dev_c);

    // Get the ending time
    clock_t endTime = clock();

    // Calculate the elapsed time in seconds
    double deltaTime = (double) (endTime - startTime) / CLOCKS_PER_SEC;

    // Print the delta time
    printf("Vector size %d - Delta time gpu: %lf seconds\n", N, deltaTime);

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy(c, dev_c, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "add kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        // return 1;
    }

    startTime = clock();

    // display the results
    for (i=0; i < N; i++) {
	c[i] = a[i] + b[i];
    }

    endTime = clock();

    deltaTime = (double) (endTime - startTime) / CLOCKS_PER_SEC;

    // Print the delta time
    printf("Vector size %d - Delta time cpu: %lf seconds\n", N, deltaTime);

    // free the memory allocated on the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // free the memory allocated on the CPU
    free(a);
    free(b);
    free(c);

    return 0;
}


