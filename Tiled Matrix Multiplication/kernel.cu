
#include "common.h"
#include "timer.h"

#define TILE_DIM 32

__global__ void mm_tiled_kernel(float *A, float *B, float *C, unsigned int M, unsigned int N, unsigned int K)
{

    // TODO
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (unsigned int tile = 0; tile < (N + TILE_DIM - 1)/TILE_DIM; ++tile)
        {

            A_s[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_DIM + threadIdx.x];
            B_s[threadIdx.y][threadIdx.x] = B[(tile * TILE_DIM + threadIdx.y) * N + col];
            __syncthreads();

            for (unsigned int i = 0; i < TILE_DIM; ++i)
            {
                sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
            }
            __syncthreads();
        }
        C[row * N + col] = sum;
    }
}

void mm_gpu(float *A, float *B, float *C, unsigned int M, unsigned int N, unsigned int K)
{

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU memory
    cudaEventRecord(start);

    // TODO
    float *A_d;
    float *B_d;
    float *C_d;
    cudaMalloc((void **)&A_d, M * N * sizeof(float));
    cudaMalloc((void **)&B_d, N * K * sizeof(float));
    cudaMalloc((void **)&C_d, M * K * sizeof(float));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Allocation time: %.3f ms\n", elapsedTime);

    // Copy data to GPU
    cudaEventRecord(start);

    // TODO
    cudaMemcpy(A_d, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy to GPU time: %.3f ms\n", elapsedTime);

    // Call kernel
    cudaEventRecord(start);

    // TODO
    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks((K + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                   (M + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    mm_tiled_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\033[1;32mKernel time: %.3f ms\033[0m\n", elapsedTime);

    // Copy data from GPU
    cudaEventRecord(start);

    // TODO
    cudaMemcpy(C, C_d, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy from GPU time: %.3f ms\n", elapsedTime);

    // Free GPU memory
    cudaEventRecord(start);

    // TODO
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);  
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Deallocation time: %.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
