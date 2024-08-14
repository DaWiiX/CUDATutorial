#include<cuda.h>
#include "cuda_runtime.h"
#include<bits/stdc++.h>

__global__ void reduce_baseline(const int* input, int* output, size_t n) {
    int sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += input[i];
    }
    *output = sum;
}

int main() {
    const int N = 25600000;
    // 设置grid和block大小
    dim3 gridsize(1, 1, 1);
    dim3 blocksize(1, 1, 1);

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    float milliseconds = 0;

    // 申请内存
    int *h_a = (int*)malloc(N * sizeof(int));
    int *h_output = (int*)malloc(sizeof(int));
    int *d_a, *d_output;
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_output, sizeof(int));

    // initialize data
    for (size_t i = 0; i < N; ++i) {
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
    reduce_baseline<<<gridsize, blocksize>>>(d_a, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device
    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    
    // check result
    if (*h_output == ground_truth) {
        printf("right!\n");
    } else {
        printf("wrong!\n");
    }

    printf("time: %f ms.\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_output);
    free(h_a);
    free(h_output);



}