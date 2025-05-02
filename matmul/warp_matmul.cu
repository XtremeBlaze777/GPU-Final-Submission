#include <cassert>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
using namespace std;

#define TILE_WIDTH 16 // for your shared‐memory version
#define WARP_ROWS 4   // warp‐tile rows  (4×8 = 32 threads)
#define WARP_COLS 8   // warp‐tile cols
#define WARP_SIZE 32
#define WARP_TILE_WIDTH 8

// ceil division helper
// inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

void verify_result(unsigned long *result, unsigned long *C, int width) {
  for (int i = 0; i < width * width; i++) {
    if (result[i] != C[i]) {
      printf("Not equal at %d: %lu vs %lu\n", i, result[i], C[i]);
      return;
    }
  }
  printf("Verification done. Result on GPU and CPU same.\n");
}

void cpu_mm(int *A, int *B, unsigned long *result, int width) {
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
  double elapsed =
      (end.tv_sec - begin.tv_sec) * 1e3 + (end.tv_usec - begin.tv_usec) * 1e-3;
  printf("Time measured on CPU: %.6f ms.\n", elapsed);
}

__global__ void warp_mm(int *A, int *B, unsigned long *C, int width) {
  __shared__ int tileA[4][8][8];
  __shared__ int tileB[4][8][8];
  int warp = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
  int lane = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
  int warpY = lane / 8;
  int warpX = lane - warpY * 8;
  int local_warp_rowY = warp / 2;
  int local_warp_rowX = warp % 2;

  int warp_row = 2 * blockIdx.y * blockDim.y + local_warp_rowY * 8 + warpY;
  int warp_col = blockIdx.x * blockDim.x + local_warp_rowX * 8 + warpX;

  unsigned long value = 0;
  unsigned long value2 = 0;
  int tiles_per_row = (width + 7) / WARP_TILE_WIDTH;
  for (int tile = 0; tile < tiles_per_row; ++tile) {
    // Load from A
    int aRow = warp_row;
    int aCol = tile * 8 + warpX;
    int aIndex = aRow * width + aCol;
    tileA[warp][warpY][warpX] = A[aIndex];

    aIndex = (aRow + 4) * width + aCol;
    tileA[warp][warpY + 4][warpX] = A[aIndex];

    // Load from B
    int bRow = tile * 8 + warpY;
    int bCol = warp_col;
    int bIndex = bRow * width + bCol;
    tileB[warp][warpY][warpX] = B[bIndex];

    bIndex = (bRow + 4) * width + bCol;
    tileB[warp][warpY + 4][warpX] = B[bIndex];

    __syncwarp();

    // Compute partial sum
    for (int k = 0; k < WARP_TILE_WIDTH; ++k) {
      unsigned long aElem = tileA[warp][warpY][k];
      unsigned long bElem = tileB[warp][k][warpX];
      value += aElem * bElem;

      unsigned long aElem2 = tileA[warp][warpY + 4][k];
      unsigned long bElem2 = tileB[warp][k][warpX];
      value2 += aElem2 * bElem2;
    }

    __syncwarp();
  }

  // Store result in C
  int cIndex = warp_row * width + warp_col;
  C[cIndex] = value;

  // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 &&
  //     threadIdx.y == 3) {
  //   printf("A[0]=%d, B[0]=%d\n", A[0], B[0]);
  //   printf("warpX=%d, warpY=%d\n", warpX, warpY);
  //   printf("value1=%ld, value2=%ld\n", value, value2);
  //   printf("cIndex=%d, warp_row=%d, warp_col=%d\n", cIndex, warp_row,
  //   warp_col); printf("myWarp=%d\n", warp);
  // }

  cIndex = (warp_row + 4) * width + warp_col;
  C[cIndex] = value2;
}
__global__ void block_mm(int *A, int *B, unsigned long *C, int width) {
  __shared__ int tileAs[TILE_WIDTH][TILE_WIDTH];
  __shared__ int tileBs[TILE_WIDTH][TILE_WIDTH];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // target element coordinates
  int row = by * TILE_WIDTH + ty;
  int column = bx * TILE_WIDTH + tx;

  int tiles_per_row = ceilf(width / (float)TILE_WIDTH);

  unsigned long pValue = 0;

  for (int i = 0; i < tiles_per_row; i++) {
    // move the tiles and update shared memory value for new tile positions
    if (row < width && (i * TILE_WIDTH + tx) < width)
      tileAs[ty][tx] = A[row * width + i * TILE_WIDTH + tx];
    else
      tileAs[ty][tx] = 0;
    if (column < width && (i * TILE_WIDTH + ty) < width)
      tileBs[ty][tx] = B[(i * TILE_WIDTH + ty) * width + column];
    else
      tileBs[ty][tx] = 0;

    // after the entire tile's values are available, proceed
    __syncthreads();

    for (int j = 0; j < TILE_WIDTH; j++)
      pValue += tileAs[ty][j] * tileBs[j][tx];
    // after the entire tile's values have been used, proceed
    __syncthreads();
  }
  // boundary check
  if (row < width && column < width)
    C[row * width + column] = pValue;
}
__global__ void thread_mm(int *A, int *B, unsigned long *C, int width) {
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
void block_runner(int *A, int *B, int width, unsigned long *cpu_res) {
  dim3 dimGrid((width + (TILE_WIDTH - 1)) / TILE_WIDTH,
               (width + (TILE_WIDTH - 1)) / TILE_WIDTH, 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  unsigned long *C =
      (unsigned long *)malloc((width * width) * sizeof(unsigned long));
  int *gpuA, *gpuB;
  unsigned long *gpuC;
  cudaMalloc(&gpuA, width * width * sizeof(int));
  cudaMalloc(&gpuB, width * width * sizeof(int));
  cudaMalloc(&gpuC, width * width * sizeof(unsigned long));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  cudaEventRecord(start, 0);
  cudaMemcpy(gpuA, A, width * width * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpuB, B, width * width * sizeof(int), cudaMemcpyHostToDevice);

  block_mm<<<dimGrid, dimBlock>>>(gpuA, gpuB, gpuC, width);

  // Copy the result matrix ’C ’ from device to host
  cudaMemcpy(C, gpuC, width * width * sizeof(unsigned long),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaDeviceSynchronize();
  printf("Block Level: %.6f ms\n", milliseconds);
  cudaFree(gpuA);
  cudaFree(gpuB);
  cudaFree(gpuC);
  verify_result(cpu_res, C, width);
  free(C);
}
void warp_runner(int *A, int *B, int width, unsigned long *cpu_res) {
  unsigned long *C =
      (unsigned long *)malloc(width * width * sizeof(unsigned long));
  int *gpuA, *gpuB;
  unsigned long *gpuC;
  cudaMalloc(&gpuA, width * width * sizeof(int));
  cudaMalloc(&gpuB, width * width * sizeof(int));
  cudaMalloc(&gpuC, width * width * sizeof(unsigned long));

  cudaMemcpy(gpuA, A, width * width * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpuB, B, width * width * sizeof(int), cudaMemcpyHostToDevice);

  dim3 dimGrid((width + (TILE_WIDTH - 1)) / TILE_WIDTH,
               (width + (TILE_WIDTH - 1)) / TILE_WIDTH, 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH / 2, 1);
  // dim3 dimBlock(WARP_SIZE, 1, 1);
  // dim3 dimGrid(iDivUp(width, WARP_COLS), iDivUp(width, WARP_ROWS), 1);

  cudaEvent_t start, stop;
  float ms = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  warp_mm<<<dimGrid, dimBlock>>>(gpuA, gpuB, gpuC, width);
  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess)
  //   printf("CUDA error: %s\n", cudaGetErrorString(err));
  // else
  //   printf("???????????\n");
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  printf("Time taken by warp‐level KERNEL: %.6f ms\n", ms);

  cudaMemcpy(C, gpuC, width * width * sizeof(unsigned long),
             cudaMemcpyDeviceToHost);
  // verify_result(cpu_res, C, width);

  cudaFree(gpuA);
  cudaFree(gpuB);
  cudaFree(gpuC);
  free(C);
}

int main(int argc, char *argv[]) {
  int width = 4096;
  if (argc == 2) {
    width = atoi(argv[1]);
  }
  printf("Matrix width: %d\n", width);

  // allocate & init host matrices
  int *A = (int *)malloc(width * width * sizeof(int));
  int *B = (int *)malloc(width * width * sizeof(int));
  for (int i = 0; i < width * width; i++) {
    A[i] = rand() % 100;
    B[i] = rand() % 100;
  }
  unsigned long *cpuRes =
      (unsigned long *)malloc(width * width * sizeof(unsigned long));
  unsigned long *gpuRes =
      (unsigned long *)malloc(width * width * sizeof(unsigned long));

  // cpu_mm(A, B, cpuRes, width);
  // printf("CPU calculation done\n");

  // … call your gpu_mm() and gpu_shm_mm() here …

  block_runner(A, B, width, cpuRes);
  warp_runner(A, B, width, cpuRes);

  free(A);
  free(B);
  free(cpuRes);
  free(gpuRes);
  return 0;
}
