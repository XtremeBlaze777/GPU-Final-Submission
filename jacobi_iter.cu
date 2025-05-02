#include <cooperative_groups.h>
#include <cstdlib>
#include <iostream>

#define ROWS_PER_CTA 8
#define N_ROWS 512
#define WARPS 8
#define ITERATIONS 100

#define CUDA_CHECK(err)                                                        \
  if (err != cudaSuccess) {                                                    \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "           \
              << __FILE__ << ":" << __LINE__ << std::endl;                     \
    exit(EXIT_FAILURE);                                                        \
  }

void Jacobi_CPU(double *A, double *b, double *x, double *x_new) {
  int n = N_ROWS;
  double *x_cpu = (double *)malloc(N_ROWS * sizeof(double));
  for (int i = 0; i < n; i++) {
    x_cpu[i] = x[i];
  }

  for (int iter = 0; iter < ITERATIONS; ++iter) {
    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        if (j != i)
          sum += A[i * n + j] * x_cpu[j]; // A[i][j]
      }
      x_new[i] = (b[i] - sum) / A[i * n + i]; // A[i][i]
    }

    for (int i = 0; i < n; ++i) {
      x_cpu[i] = x_new[i];
    }
  }
  free(x_cpu);
}
__global__ void Jacobi_block(const double *A, const double *b, double *x,
                             double *x_new) {
  __shared__ double x_shared[N_ROWS];
  __shared__ double b_shared[ROWS_PER_CTA]; // mayb add +1

  // fill Xs

  for (int i = threadIdx.x; i < N_ROWS; i += blockDim.x) {
    x_shared[i] = x[i];
  }
  // load b vector
  // Big iteration loop

  if (threadIdx.x < ROWS_PER_CTA) { // a thread per row "assigned" to each block
    int k = threadIdx.x;
    for (int i = k + (blockIdx.x * ROWS_PER_CTA); // i = current row
         (k < ROWS_PER_CTA) && (i < N_ROWS); // keep going until off the matrix
         k += ROWS_PER_CTA, i += ROWS_PER_CTA) { // increment by
      b_shared[i % (ROWS_PER_CTA + 1)] = b[i];
    }
  }
  __syncthreads();

  for (int k = 0, i = blockIdx.x * ROWS_PER_CTA;
       (k < ROWS_PER_CTA) && (i < N_ROWS); k++, i++) {
    double rowThreadSum = 0.0;
    for (int j = threadIdx.x; j < N_ROWS; j += blockDim.x) {
      rowThreadSum += (A[i * N_ROWS + j] * x_shared[j]);
    }

    atomicAdd(&b_shared[i % (ROWS_PER_CTA + 1)], -rowThreadSum);
  }

  __syncthreads();

  if (threadIdx.x < ROWS_PER_CTA) {
    int k = threadIdx.x;

    for (int i = k + (blockIdx.x * ROWS_PER_CTA);
         (k < ROWS_PER_CTA) && (i < N_ROWS);
         k += ROWS_PER_CTA, i += ROWS_PER_CTA) {
      double dx = b_shared[i % (ROWS_PER_CTA + 1)];
      dx /= A[i * N_ROWS + i];

      x_new[i] = (x_shared[i] + dx);
    }
  }
}

static __global__ void Jacobi_warp(const double *A, const double *b, double *x,
                                   double *x_new) {
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  unsigned int mask = 0xffffffff;
  __shared__ double x_shared[WARPS][N_ROWS]; // N_ROWS == n
  __shared__ double b_shared[WARPS];         // + 1 for optimization
  int myRow = warp + WARPS * blockIdx.x;
  for (int i = lane; i < N_ROWS; i += 32) { // load xs into shared memory
    x_shared[warp][i] = x[i];
  }
  if (lane == 0) {
    b_shared[warp] = b[myRow];
  }
  __syncwarp(mask);

  double rowThreadSum = 0.0;
  for (int j = lane; j < N_ROWS; j += 32) {
    rowThreadSum += (A[myRow * N_ROWS + j] * x_shared[warp][j]);
  }
  __syncwarp(mask);
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    rowThreadSum += __shfl_down_sync(mask, rowThreadSum, offset);
  }

  if (lane == 0) {
    // b_shared[warp] -= rowThreadSum;
    double dx = b_shared[warp] - rowThreadSum;
    dx /= A[myRow * N_ROWS + myRow];
    //
    x_new[myRow] = x_shared[warp][myRow] + dx;
  }
}

static __global__ void Jacobi_thread(const double *A, const double *b,
                                     double *x, double *x_new) {
  if (threadIdx.x < N_ROWS) {
    double rowThreadSum = 0.0;
    for (int j = 0; j < N_ROWS; j++) {
      rowThreadSum += (A[threadIdx.x * N_ROWS + j] * x[j]);
    }

    // b_shared[warp] -= rowThreadSum;
    double dx = b[threadIdx.x] - rowThreadSum;
    dx /= A[threadIdx.x * N_ROWS + threadIdx.x];
    //
    x_new[threadIdx.x] = x[threadIdx.x] + dx;
  }
}
__global__ void Jacobi_device(const double *A, const double *b, double *x,
                              double *x_new, int mykernel,
                              int blocks_per_kernel) {
  __shared__ double x_shared[N_ROWS];
  __shared__ double b_shared; // mayb add +1

  // fill Xs
  int myRow = mykernel * blocks_per_kernel + blockIdx.x;
  if (threadIdx.x == 0)
    b_shared = b[myRow];
  for (int i = threadIdx.x; i < N_ROWS; i += blockDim.x) {
    x_shared[i] = x[i];
  }
  // load b vector
  // Big iteration loop

  __syncthreads();

  double rowThreadSum = 0.0;
  for (int j = threadIdx.x; j < N_ROWS; j += blockDim.x) {
    rowThreadSum += (A[myRow * N_ROWS + j] * x_shared[j]);
  }

  atomicAdd(&b_shared, -rowThreadSum);

  __syncthreads();

  if (threadIdx.x == 0) {

    double dx = b_shared;
    dx /= A[myRow * N_ROWS + myRow];

    x_new[myRow] = (x_shared[myRow] + dx);
  }
}

void block_runner(double *A, double *b, double *x_default, double *x_new,
                  int timed) {
  double *d_A, *d_b, *d_x, *d_x_new, *d_b_new, *tmp;
  // CUDA memory allocation
  size_t size = N_ROWS * sizeof(double);
  size_t matrix_size = N_ROWS * N_ROWS * sizeof(double);
  double *x = (double *)malloc(sizeof(double) * N_ROWS);
  for (int i = 0; i < N_ROWS; i++) {
    x[i] = x_default[i];
  }
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  CUDA_CHECK(cudaMalloc(&d_A, matrix_size));
  CUDA_CHECK(cudaMalloc(&d_b, size));
  CUDA_CHECK(cudaMalloc(&d_x, size));
  CUDA_CHECK(cudaMalloc(&d_x_new, size));
  CUDA_CHECK(cudaMalloc(&d_b_new, size));
  //  CUDA_CHECK(cudaMalloc(&d_diff, size));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_A, A, matrix_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_new, d_b_new, size, cudaMemcpyHostToDevice));

  dim3 block_size(256, 1, 1);
  dim3 block_count(64, 1, 1);
  for (int iter = 0; iter < ITERATIONS; iter++) {
    Jacobi_block<<<block_count, block_size>>>(d_A, d_b, d_x, d_x_new);
    tmp = d_x;
    d_x = d_x_new;
    d_x_new = tmp;
  }

  CUDA_CHECK(cudaMemcpy(x_new, d_x, size, cudaMemcpyDeviceToHost));
  cudaEventRecord(end, 0);
  float ms = 0;
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, start, end);
  if (timed)
    printf("block TIMING: %f\n", ms);

  cudaFree(d_A);
  cudaFree(d_b);
  cudaFree(d_x);
  cudaFree(d_x_new);
  free(x);
}
void warp_runner(double *A, double *b, double *x_default, double *x_new,
                 int timed) {
  double *d_A, *d_b, *d_x, *d_x_new, *d_b_new, *tmp;
  // CUDA memory allocation
  size_t size = N_ROWS * sizeof(double);
  size_t matrix_size = N_ROWS * N_ROWS * sizeof(double);
  double *x = (double *)malloc(sizeof(double) * N_ROWS);
  for (int i = 0; i < N_ROWS; i++) {
    x[i] = x_default[i];
  }
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  CUDA_CHECK(cudaMalloc(&d_A, matrix_size));
  CUDA_CHECK(cudaMalloc(&d_b, size));
  CUDA_CHECK(cudaMalloc(&d_x, size));
  CUDA_CHECK(cudaMalloc(&d_x_new, size));
  CUDA_CHECK(cudaMalloc(&d_b_new, size));
  //  CUDA_CHECK(cudaMalloc(&d_diff, size));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_A, A, matrix_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_new, d_b_new, size, cudaMemcpyHostToDevice));

  dim3 block_size(256, 1, 1);
  dim3 block_count(64, 1, 1);
  for (int iter = 0; iter < ITERATIONS; iter++) {
    Jacobi_warp<<<block_count, block_size>>>(d_A, d_b, d_x, d_x_new);
    tmp = d_x;
    d_x = d_x_new;
    d_x_new = tmp;
  }

  CUDA_CHECK(cudaMemcpy(x_new, d_x, size, cudaMemcpyDeviceToHost));
  cudaEventRecord(end, 0);
  float ms = 0;
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, start, end);
  if (timed)
    printf("warp TIMING: %f\n", ms);

  cudaFree(d_A);
  cudaFree(d_b);
  cudaFree(d_x);
  cudaFree(d_x_new);
  free(x);
}

void thread_runner(double *A, double *b, double *x_default, double *x_new,
                   int timed) {
  double *d_A, *d_b, *d_x, *d_x_new, *d_b_new, *tmp;
  // CUDA memory allocation
  size_t size = N_ROWS * sizeof(double);
  size_t matrix_size = N_ROWS * N_ROWS * sizeof(double);
  double *x = (double *)malloc(sizeof(double) * N_ROWS);
  for (int i = 0; i < N_ROWS; i++) {
    x[i] = x_default[i];
  }

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  CUDA_CHECK(cudaMalloc(&d_A, matrix_size));
  CUDA_CHECK(cudaMalloc(&d_b, size));
  CUDA_CHECK(cudaMalloc(&d_x, size));
  CUDA_CHECK(cudaMalloc(&d_x_new, size));
  CUDA_CHECK(cudaMalloc(&d_b_new, size));
  //  CUDA_CHECK(cudaMalloc(&d_diff, size));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_A, A, matrix_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_new, d_b_new, size, cudaMemcpyHostToDevice));

  dim3 block_size(256, 1, 1);
  dim3 block_count(64, 1, 1);
  for (int iter = 0; iter < ITERATIONS; iter++) {
    Jacobi_thread<<<block_count, block_size>>>(d_A, d_b, d_x, d_x_new);
    tmp = d_x;
    d_x = d_x_new;
    d_x_new = tmp;
  }

  CUDA_CHECK(cudaMemcpy(x_new, d_x, size, cudaMemcpyDeviceToHost));
  cudaEventRecord(end, 0);
  float ms = 0;
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, start, end);

  if (timed)
    printf("thread TIMING: %f\n", ms);

  cudaFree(d_A);
  cudaFree(d_b);
  cudaFree(d_x);
  cudaFree(d_x_new);
  free(x);
}
void device_runner(double *A, double *b, double *x_default, double *x_new,
                   int timed, int device_slices) {
  double *d_A, *d_b, *d_x, *d_x_new, *d_b_new, *tmp;
  // CUDA memory allocation
  size_t size = N_ROWS * sizeof(double);
  size_t matrix_size = N_ROWS * N_ROWS * sizeof(double);
  double *x = (double *)malloc(sizeof(double) * N_ROWS);
  for (int i = 0; i < N_ROWS; i++) {
    x[i] = x_default[i];
  }

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  CUDA_CHECK(cudaMalloc(&d_A, matrix_size));
  CUDA_CHECK(cudaMalloc(&d_b, size));
  CUDA_CHECK(cudaMalloc(&d_x, size));
  CUDA_CHECK(cudaMalloc(&d_x_new, size));
  CUDA_CHECK(cudaMalloc(&d_b_new, size));
  //  CUDA_CHECK(cudaMalloc(&d_diff, size));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_A, A, matrix_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_new, d_b_new, size, cudaMemcpyHostToDevice));

  if (device_slices == 0) {
    device_slices = N_ROWS;
  }
  int blocks_per_kernel = (N_ROWS + device_slices - 1) / device_slices;
  dim3 block_size(256, 1, 1);
  dim3 block_count(blocks_per_kernel, 1, 1);
  cudaStream_t *streams =
      (cudaStream_t *)malloc(device_slices * sizeof(cudaStream_t));
  for (int i = 0; i < device_slices; i++) {
    cudaStreamCreate(&streams[i]);
  }
  for (int iter = 0; iter < ITERATIONS; iter++) {
    for (int i = 0; i < device_slices; i++) {
      Jacobi_device<<<block_count, block_size, 0, streams[i]>>>(
          d_A, d_b, d_x, d_x_new, i, blocks_per_kernel);
    }
    cudaDeviceSynchronize();
    tmp = d_x;
    d_x = d_x_new;
    d_x_new = tmp;
  }

  CUDA_CHECK(cudaMemcpy(x_new, d_x, size, cudaMemcpyDeviceToHost));
  cudaEventRecord(end, 0);
  float ms = 0;
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, start, end);

  if (timed)
    printf("Device TIMING: %f\n", ms);

  cudaFree(d_A);
  cudaFree(d_b);
  cudaFree(d_x);
  cudaFree(d_x_new);
  free(x);
}
int main(int argc, char *argv[]) {
  double *A, *b, *x, *x_new, *x_new_CPU;

  size_t size = N_ROWS * sizeof(double);
  size_t matrix_size = N_ROWS * N_ROWS * sizeof(double);

  int device_slices = 0;
  if (argc > 1) {
    device_slices = atoi(argv[1]);
  }
  printf("dev Slices = %d\n", device_slices);

  // Host allocations
  A = (double *)malloc(matrix_size);
  b = (double *)malloc(size);
  x = (double *)malloc(size);
  x_new = (double *)malloc(size);
  x_new_CPU = (double *)malloc(size);
  // diff = (double *)malloc(size);

  // Initialize A (diagonally dominant), b, and x
  for (int i = 0; i < N_ROWS; ++i) {
    b[i] = 1.0;
    x[i] = 0.0;
    x_new_CPU[i] = 0.0;
    for (int j = 0; j < N_ROWS; ++j) {
      A[i * N_ROWS + j] = (i == j) ? (double)(N_ROWS) : 1;
    }
  }

  struct timespec h_begin, h_end;
  // calculate vector addition on CPU
  double cpu_time = 0.0;
  // for (int i = 0; i < 100; i++) {
  clock_gettime(CLOCK_REALTIME, &h_begin);
  Jacobi_CPU(A, b, x, x_new_CPU);
  // block_runner(A, b, x, x_new);
  clock_gettime(CLOCK_REALTIME, &h_end);
  cpu_time += (h_end.tv_sec - h_begin.tv_sec) * 1e3 +
              (h_end.tv_nsec - h_begin.tv_nsec) * 1e-6;
  // }

  printf("CPU TIMING: %lf\n", cpu_time / 100);
  // for (int i = 0; i < 50; i++) {
  //   printf("CPU[%d]=%.20f, kernel[%d]=%.20f\n", i, x_new_CPU[i], i,
  //   x_new[i]); if (x_new_CPU[i] != x_new[i]) {
  //
  //     std::cerr << "kernel result incorrect" << std::endl;
  //     exit(1);
  //   }
  // }

  // for (int i = 0; i < 50; i++) {
  //   printf("CPU[%d]=%f, kernel[%d]=%f\n", i, x_new_CPU[i], i, x_new[i]);
  //   if (x_new_CPU[i] != x_new[i]) {
  //
  //     std::cerr << "kernel result incorrect" << std::endl;
  //     exit(1);
  //   }
  // }
  // for (int i = 0; i < 50; i++) {
  //   printf("CPU[%d]=%.20f, kernel[%d]=%.20f\n", i, x_new_CPU[i], i,
  //   x_new[i]); if (x_new_CPU[i] != x_new[i]) {
  //
  //     std::cerr << "kernel result incorrect" << std::endl;
  //     exit(1);
  //   }
  // }

  warp_runner(A, b, x, x_new, 0);
  device_runner(A, b, x, x_new, 0, device_slices);
  thread_runner(A, b, x, x_new, 0);
  block_runner(A, b, x, x_new, 0);

  block_runner(A, b, x, x_new, 1);
  warp_runner(A, b, x, x_new, 1);
  device_runner(A, b, x, x_new, 1, device_slices);
  for (int i = 0; i < N_ROWS; i++) {
    // printf("CPU[%d]=%.20f, kernel[%d]=%.20f\n", i, x_new_CPU[i], i,
    // x_new[i]);
    float diff = fabs(x_new_CPU[i] - x_new[i]);
    if (diff > 1e-3) {

      std::cerr << "kernel result incorrect" << std::endl;
      exit(1);
    }
  }
  thread_runner(A, b, x, x_new, 1);
  // Print first few values
  // std::cout << "Final solution (first 10 values):" << std::endl;
  // for (int i = 0; i < 10; ++i) {
  //   std::cout << x_new_CPU[i] << " ";
  // }
  // std::cout << std::endl;

  // Cleanup
  free(A);
  free(b);
  free(x);
  free(x_new);
  free(x_new_CPU);

  return 0;
}
