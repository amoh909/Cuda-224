
#include "common.h"
#include "timer.h"

__global__ void daxpy_kernel(double *x, double *y, double a, unsigned int M)
{

    // TODO
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < M)
    {
        y[i] = a * x[i] + y[i];
    }
}

void daxpy_gpu(double *x, double *y, double a, unsigned int M)
{

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU memory
    cudaEventRecord(start);

    // TODO
    // No need to allocate for a as the GPU automatically takes care of the situation if a wasn't a pointer
    double *x_d, *y_d;
    cudaMalloc((void **)&x_d, M * sizeof(double));
    cudaMalloc((void **)&y_d, M * sizeof(double));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Allocation time: %.3f ms\n", elapsedTime);

    // Copy data to GPU
    cudaEventRecord(start);

    // TODO

    cudaMemcpy(x_d, x, M * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, M * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy to GPU time: %.3f ms\n", elapsedTime);

    // Call kernel
    cudaEventRecord(start);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    } // error checking at the kernel level

    // TODO
    const unsigned int numThreadsPerBlock = 256;
    const unsigned int numBlocks = (M + numThreadsPerBlock - 1) / numThreadsPerBlock;
    daxpy_kernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, a, M);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\033[1;32mKernel time: %.3f ms\033[0m\n", elapsedTime);

    // Copy data from GPU
    cudaEventRecord(start);

    // TODO
    cudaMemcpy(y, y_d, M * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy from GPU time: %.3f ms\n", elapsedTime);

    // Free GPU memory
    cudaEventRecord(start);

    // TODO
    cudaFree(x_d);
    cudaFree(y_d);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Deallocation time: %.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
