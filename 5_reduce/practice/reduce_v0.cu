#include <cuda.h>
#include "cuda_runtime.h"
#include <bits/stdc++.h>

// 1012.837891 ms
__global__ void reduce_baseline(const int *input, int *output, size_t n)
{
    int sum = 0;
    for (size_t i = 0; i < n; ++i)
    {
        sum += input[i];
    }
    *output = sum;
}

// 325.971924 ms
// after warm up 1.390144 ms
template <int blockSize>
__global__ void reduce_v0(const int *input, int *output, size_t n)
{
    __shared__ int smem[blockSize];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockSize + tid;
    smem[tid] = input[gid];
    __syncthreads();

    for (int depth = 1; depth < blockDim.x; depth *= 2)
    {
        if (tid % (2 * depth) == 0)
        {
            smem[tid] = smem[tid] + smem[tid + depth];
        }
        // 进入下一个循环之前需要同步
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

bool checkResults(const int *output, const int *ground_truth, int blockSize)
{
    int sum = 0;
    for (int i = 0; i < blockSize; ++i)
    {
        sum += output[i];
    }
    if (sum != *ground_truth)
    {
        printf("the result is %d, the correct result is %d.\n", sum, *ground_truth);
        return false;
    }

    else
        return true;
}

int main()
{
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    // 设置grid和block大小
    const int blockSize = 256;
    int gridSize = std::min((N - 1) / blockSize + 1, deviceProp.maxGridSize[0]);

    float milliseconds = 0;
    dim3 gridsize(gridSize, 1, 1);
    dim3 blocksize(blockSize, 1, 1);

    // 申请内存
    int *h_a = (int *)malloc(N * sizeof(int));
    int *h_output = (int *)malloc(gridSize * sizeof(int));
    int *d_a, *d_output;
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_output, gridSize * sizeof(int));

    // initialize data
    for (size_t i = 0; i < N; ++i)
    {
        h_a[i] = 1;
    }
    int ground_truth = N;
    // copy data to device
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);

    // record cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // call GPU kernel
    reduce_v0<blockSize><<<gridsize, blocksize>>>(d_a, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device
    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // check result
    bool result = checkResults(h_output, &ground_truth, gridSize);
    if (result)
    {
        printf("result is correct.\n");
    }
    else
    {
        printf("result is incorrect.\n");
    }
    printf("grid size: %d, block size: %d, time: %f ms.\n", gridSize, blockSize, milliseconds);
    printf("time: %f ms.\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_output);
    free(h_a);
    free(h_output);
}