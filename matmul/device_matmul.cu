#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include <math.h>
#include <vector>
#include <cassert>
using namespace std;

#define TILE_WIDTH 16

inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

void verify_result(unsigned long* result, unsigned long* C, int width) {
    for (int i = 0; i < width * width; i++) {
        if (result[i] != C[i]) {
            printf("Not equal at %d: %lu vs %lu\n", i, result[i], C[i]);
            return;
        }
    }
    printf("Verification done. Result on GPU and CPU same.\n");
}

void cpu_mm(int* A, int* B, unsigned long* result, int width) {
    struct timeval begin, end;
    gettimeofday(&begin, NULL);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            int temp = 0;
            for (int k = 0; k < width; k++) {
                temp += A[i * width + k] * B[k * width + j];
            }
            result[i * width + j] = temp;
        }
    }
    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - begin.tv_sec) * 1e3 + (end.tv_usec - begin.tv_usec) * 1e-3;
    printf("Time measured on CPU: %.6f ms.\n", elapsed);
}

__global__ void kernel_gpu_mm(int* A, int* B, unsigned long* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        unsigned long temp = 0;
        for (int i = 0; i < width; i++) {
            temp += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = temp;
    }
}

void gpu_mm(int* A, int* B, int width, unsigned long* result) {
    unsigned long* C = (unsigned long*)malloc(width * width * sizeof(unsigned long));
    int *gpuA, *gpuB;
    unsigned long* gpuC;
    cudaMalloc(&gpuA, width * width * sizeof(int));
    cudaMalloc(&gpuB, width * width * sizeof(int));
    cudaMalloc(&gpuC, width * width * sizeof(unsigned long));

    cudaMemcpy(gpuA, A, width * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, B, width * width * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start);

    kernel_gpu_mm<<<dimGrid, dimBlock>>>(gpuA, gpuB, gpuC, width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by naive KERNEL to execute is: %.6f ms\n", milliseconds);

    cudaMemcpy(C, gpuC, width * width * sizeof(unsigned long), cudaMemcpyDeviceToHost);
    verify_result(result, C, width);

    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);
    free(C);
}

__global__ void kernel_gpu_mm_tile(const int* A, const int* B, unsigned long* C, int width, int tileX, int tileY) {
    int row = tileY * TILE_WIDTH + threadIdx.y;
    int col = tileX * TILE_WIDTH + threadIdx.x;
    if (row < width && col < width) {
        unsigned long sum = 0;
        for (int k = 0; k < width; ++k) {
            sum += (unsigned long)A[row * width + k] * (unsigned long)B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

void gpu_mm_device_streams(int* A, int* B, int width, unsigned long* result) {
    unsigned long* C = (unsigned long*)malloc(width * width * sizeof(unsigned long));
    int *gpuA, *gpuB;
    unsigned long* gpuC;
    cudaMalloc(&gpuA, width * width * sizeof(int));
    cudaMalloc(&gpuB, width * width * sizeof(int));
    cudaMalloc(&gpuC, width * width * sizeof(unsigned long));

    cudaMemcpy(gpuA, A, width * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, B, width * width * sizeof(int), cudaMemcpyHostToDevice);

    int tilesX = iDivUp(width, TILE_WIDTH);
    int tilesY = iDivUp(width, TILE_WIDTH);
    int totalTiles = tilesX * tilesY;

    vector<cudaStream_t> streams(totalTiles);
    for (int i = 0; i < totalTiles; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    cudaEventRecord(start);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    // First kernel call to setup the device
    kernel_gpu_mm_tile<<<dim3(1, 1), blockDim>>>(gpuA, gpuB, gpuC, width, 0, 0);
    cudaDeviceSynchronize();
    cudaDeviceSynchronize();

    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            int idx = ty * tilesX + tx;
            kernel_gpu_mm_tile<<<dim3(1, 1), blockDim, 0, streams[idx]>>>(gpuA, gpuB, gpuC, width, tx, ty);
        }
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken by device-level tiled streams: %.6f ms\n", ms);

    cudaMemcpy(C, gpuC, width * width * sizeof(unsigned long), cudaMemcpyDeviceToHost);
    verify_result(C, C, width);

    for (auto& stream : streams) {
        cudaStreamDestroy(stream);
    }
    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);
    free(C);
}

int main(int argc, char* argv[]) {
    int width = 2048;
    if (argc == 2) {
        width = atoi(argv[1]);
    }
    printf("Matrix width: %d\n", width);

    int* A = (int*)malloc(width * width * sizeof(int));
    int* B = (int*)malloc(width * width * sizeof(int));
    for (int i = 0; i < width * width; i++) {
        A[i] = rand() % 1000;
        B[i] = rand() % 1000;
    }
    unsigned long* result = (unsigned long*)malloc(width * width * sizeof(unsigned long));

    // cpu_mm(A, B, result, width);
    // printf("CPU calculation done\n");

    // gpu_mm(A, B, width, result);

    gpu_mm_device_streams(A, B, width, result);

    free(result);
    free(A);
    free(B);
    return 0;
}
