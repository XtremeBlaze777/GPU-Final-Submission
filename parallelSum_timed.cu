#include <cooperative_groups.h>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <time.h>

__global__ void parallelSum_thread(int *inputArray, int *outputResult,
                                   int arraySize) {
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  int myData = (globalID < arraySize) ? inputArray[globalID] : 0;
  atomicAdd(&outputResult[0], myData);
}
__global__ void parallelSum_warp(int *inputArray, int *outputResult,
                                 int arraySize) {
  int lane = threadIdx.x % 32;
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int mask = 0xffffffff;
  // if (threadIdx.x == 0) {
  //   outputResult[0] = 0;
  // }
  //__syncthreads();
  int myData = (globalID < arraySize) ? inputArray[globalID] : 0;
  __syncwarp(mask);
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    myData += __shfl_down_sync(mask, myData, offset);
  }
  __syncwarp(mask);

  if (lane == 0) {
    atomicAdd(&outputResult[0], myData);
  }
}

__global__ void parallelSum_device(int *inputArray, int *outputResult,
                                   int arraySize, int myBlock) {
  extern __shared__ int sharedMemory[];
  int threadID = threadIdx.x;
  int globalID = myBlock * blockDim.x + threadIdx.x;

  sharedMemory[threadID] = (globalID < arraySize) ? inputArray[globalID] : 0;
  __syncthreads();
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * threadID;
    if (index < blockDim.x) {
      sharedMemory[index] += sharedMemory[index + stride];
    }
    __syncthreads();
  }

  if (threadID == 0) {
    atomicAdd(&outputResult[myBlock], sharedMemory[0]);
  }
}
__global__ void parallelSum_block(int *inputArray, int *outputResult,
                                  int arraySize) {
  extern __shared__ int sharedMemory[];
  int threadID = threadIdx.x;
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  // if (threadID == 0) {
  //   *outputResult = 0;
  // }
  sharedMemory[threadID] = (globalID < arraySize) ? inputArray[globalID] : 0;
  __syncthreads();
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * threadID;
    if (index < blockDim.x) {
      sharedMemory[index] += sharedMemory[index + stride];
    }
    __syncthreads();
  }

  if (threadID == 0) {
    atomicAdd(&outputResult[0], sharedMemory[0]);
  }
}
__global__ void parallelSum_kernel(int *inputArray, int *outputResult,
                                   int arraySize) {
  extern __shared__ int sharedMemory[];
  int threadID = threadIdx.x;
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  sharedMemory[threadID] = (globalID < arraySize) ? inputArray[globalID] : 0;
  __syncthreads();
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * threadID;
    if (index < blockDim.x) {
      sharedMemory[index] += sharedMemory[index + stride];
    }
    __syncthreads();
  }

  if (threadID == 0) {
    outputResult[blockIdx.x] = sharedMemory[0];
  }
}

void CPUSum(int *inputArray, int *outputResult, int arraySize) {
  int sum = 0;
  for (int i = 0; i < arraySize; i++) {
    sum += inputArray[i];
  }
  *outputResult = sum;
}

void parallel_block(int *A, int *B, int width, int timed) {
  int size = width * sizeof(int);
  int *d_A, *d_B;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, sizeof(int));
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(int), cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  int threads_per_block = 1024;
  int block_count = (width + threads_per_block) / threads_per_block;
  dim3 dimGrid(block_count, 1, 1);
  dim3 dimBlock(threads_per_block, 1, 1);
  parallelSum_block<<<dimGrid, dimBlock, size / block_count>>>(d_A, d_B, width);
  cudaMemcpy(B, d_B, sizeof(int), cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  float ms = 0;
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, start, end);

  if (timed)
    printf("BLOCK TIMING: %f\n", ms);
  cudaFree(d_A);
  cudaFree(d_B);
}
void parallel_warp(int *A, int *B, int width, int timed) {
  int size = width * sizeof(int);
  int *d_A, *d_B;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, sizeof(int));
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(int), cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  int block_count = (width + 255) / 256;
  dim3 dimGrid(block_count, 1, 1);
  dim3 dimBlock(256, 1, 1);
  parallelSum_warp<<<dimGrid, dimBlock, size / block_count>>>(d_A, d_B, width);
  cudaMemcpy(B, d_B, sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  float ms = 0;
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, start, end);

  if (timed)
    printf("warp TIMING: %f\n", ms);
  cudaFree(d_A);
  cudaFree(d_B);
}
void parallel_thread(int *A, int *B, int width, int timed) {
  int size = width * sizeof(int);
  int *d_A, *d_B;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, sizeof(int));
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(int), cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  int block_count = (width + 255) / 256;
  dim3 dimGrid(block_count, 1, 1);
  dim3 dimBlock(256, 1, 1);
  parallelSum_thread<<<dimGrid, dimBlock, size / block_count>>>(d_A, d_B,
                                                                width);
  cudaMemcpy(B, d_B, sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  float ms = 0;
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, start, end);

  if (timed)
    printf("thread TIMING: %f\n", ms);
  cudaFree(d_A);
  cudaFree(d_B);
}
#define BLOCKS ((100000000 + 255) / 256)
void parallel_device(int *A, int *B, int width, int timed) {
  int size = width * sizeof(int);
  int *d_A, *d_B;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, sizeof(int) * BLOCKS);
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemset(d_B, 0, sizeof(int) * BLOCKS);
  cudaDeviceSynchronize();

  // Define grid and block dimensions
  dim3 dimBlock(256, 1, 1);
  cudaStream_t streams[BLOCKS];
  for (int i = 0; i < BLOCKS; i++) {
    cudaStreamCreate(&streams[i]);
  }
  for (int i = 0; i < BLOCKS; i++) {
    parallelSum_device<<<1, dimBlock, 256 * sizeof(int), streams[i]>>>(
        d_A, d_B, width, i);
  }
  cudaDeviceSynchronize();
  cudaMemcpy(B, d_B, sizeof(int) * BLOCKS, cudaMemcpyDeviceToHost);
  for (int i = 1; i < BLOCKS; i++) {

    B[0] += B[i];
  }
  cudaEventRecord(end, 0);
  float ms = 0;
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, start, end);

  if (timed)
    printf("Device TIMING: %f\n", ms);
  cudaFree(d_A);
  cudaFree(d_B);
}
int main() {
  const int width = 100000000;
  int *A = (int *)malloc(width * sizeof(int));
  int *out_GPU_multi = (int *)malloc(width * sizeof(int));
  int out_GPU[1];
  int out_CPU = 0;

  for (int i = 0; i < width; i++) {
    A[i] = rand() % 100;
    // A[i] = 1;
  }
  // Calculate result on CPU
  struct timespec h_begin, h_end;
  // calculate vector addition on CPU
  double cpu_time = 0.0;
  // for (int i = 0; i < 100; i++) {
  clock_gettime(CLOCK_REALTIME, &h_begin);
  CPUSum(A, &out_CPU, width);
  clock_gettime(CLOCK_REALTIME, &h_end);
  cpu_time += (h_end.tv_sec - h_begin.tv_sec) * 1e3 +
              (h_end.tv_nsec - h_begin.tv_nsec) * 1e-6;
  // }

  printf("CPU TIMING: %lf\n", cpu_time);

  out_GPU[0] = 0;
  parallel_warp(A, out_GPU, width, 0);
  // if (out_CPU != out_GPU[0]) {
  //   printf("CPU = %d, GPU = %d\n", out_CPU, out_GPU[0]);
  //   std::cerr << "warp result incorrect" << std::endl;
  //   exit(1);
  // }

  out_GPU[0] = 0;
  parallel_thread(A, out_GPU, width, 0);
  // if (out_CPU != out_GPU[0]) {
  //   printf("CPU = %d, GPU = %d\n", out_CPU, out_GPU[0]);
  //   std::cerr << "thread result incorrect" << std::endl;
  //   exit(1);
  // }
  out_GPU[0] = 0;
  parallel_block(A, out_GPU, width, 0);
  // if (out_CPU != out_GPU[0]) {
  //   printf("CPU = %d, GPU = %d\n", out_CPU, out_GPU[0]);
  //   std::cerr << "parallel result incorrect" << std::endl;
  //   exit(1);
  // }
  //
  out_GPU[0] = 0;
  parallel_device(A, out_GPU_multi, width, 0);
  // if (out_CPU != out_GPU_multi[0]) {
  //   printf("CPU = %d, GPU = %d\n", out_CPU, out_GPU_multi[0]);
  //   std::cerr << "device result incorrect" << std::endl;
  //   exit(1);
  // }
  //

  out_GPU[0] = 0;
  parallel_warp(A, out_GPU, width, 1);
  out_GPU[0] = 0;
  parallel_thread(A, out_GPU, width, 1);
  out_GPU[0] = 0;
  parallel_block(A, out_GPU, width, 1);
  out_GPU[0] = 0;
  parallel_device(A, out_GPU_multi, width, 1);
  free(A);
  return 0;
}
