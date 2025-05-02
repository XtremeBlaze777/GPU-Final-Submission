#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <vector>

__global__ void device_tile_histogram(int* A, int* out, int N, int offset, int chunkSize) {
    int tid = threadIdx.x;
    int gid = offset + tid;

    __shared__ int local[256];
    if (tid < 256) {
        local[tid] = 0;
    }
    __syncthreads();

    if (gid < N) {
        atomicAdd(&local[A[gid]], 1);
    }
    __syncthreads();

    if (tid < 256) {
        atomicAdd(&out[tid], local[tid]);
    }
}

void run_device_level_histogram(int* A, int* result, int N) {
    int* A_d;
    int* out_d;
    cudaMalloc(&A_d, N * sizeof(int));
    cudaMalloc(&out_d, 256 * sizeof(int));

    cudaMemcpy(A_d, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(out_d, 0, 256 * sizeof(int));

    const int chunkSize = 1024;
    const int threadsPerBlock = 1024;
    const int numChunks = (N + chunkSize - 1) / chunkSize;

    std::vector<cudaStream_t> streams(numChunks);
    for (int i = 0; i < numChunks; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch dummy kernel to setup device
    device_tile_histogram<<<1, threadsPerBlock>>>(A_d, out_d, N, 0, chunkSize);
    cudaDeviceSynchronize();
    cudaMemset(out_d, 0, 256 * sizeof(int));
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < numChunks; ++i) {
        int offset = i * chunkSize;
        device_tile_histogram<<<1, threadsPerBlock, 0, streams[i]>>>(A_d, out_d, N, offset, chunkSize);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    // printf("Time taken by device-level (streamed) histogram: %.6f ms\n", ms);
	printf("%f\n", ms);
	fflush(stdout);

    cudaMemcpy(result, out_d, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    for (auto& stream : streams) {
        cudaStreamDestroy(stream);
    }
    cudaFree(A_d);
    cudaFree(out_d);
}

void serial_histogram(int* A, int N, int* result) {
    for (int i = 0; i < N; i++) {
        result[A[i]]++;
    }
}

void compare_results(int* result, int* out) {
    for (int i = 0; i < 256; i++) {
        if (result[i] != out[i]) {
            fprintf(stderr, "incorrect logic on GPU at bin %d: CPU=%d, GPU=%d\n", i, result[i], out[i]);
            return;
        }
    }
    // printf("Correctness check done!\n");
}

int main(int argc, char* argv[]) {
    int N = 1000000;
    if (argc == 2) {
        N = atoi(argv[1]);
    }
    // printf("Vector length : %d\n", N);

    size_t bytes = sizeof(int) * N;
    int* A = (int*)malloc(bytes);
    int* result = (int*)malloc(256 * sizeof(int));
    int* out = (int*)malloc(256 * sizeof(int));

    for (int i = 0; i < N; i++) {
        A[i] = rand() % 256;
    }
    for (int i = 0; i < 256; i++) {
        result[i] = 0;
    }

    struct timespec begin, end;
    clock_gettime(CLOCK_REALTIME, &begin);
    serial_histogram(A, N, result);
    clock_gettime(CLOCK_REALTIME, &end);
    double elapsed_msec = (end.tv_sec - begin.tv_sec) * 1e3 + (end.tv_nsec - begin.tv_nsec) * 1e-6;
    // printf("Elapsed time on CPU: %.6f ms.\n", elapsed_msec);

    run_device_level_histogram(A, out, N);
    compare_results(result, out);

    free(A);
    free(result);
    free(out);
    return 0;
}
