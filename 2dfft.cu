#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define WIDTH 1024
#define HEIGHT 1024
#define WARPS 4

const float PI = 3.14159265358979323846f;

__device__ __host__ inline float2 complexMul(float2 a, float2 b) {
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __host__ inline float2 complexAdd(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __host__ inline float2 complexSub(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __host__ inline int bit_reverse(int x, int logn) {
  int result = x;
  result = ((result & 0x55555555) << 1) |
           ((result >> 1) & 0x55555555); // Swap adjacent bits
  result = ((result & 0x33333333) << 2) |
           ((result >> 2) & 0x33333333); // Swap 2-bit pairs
  result = ((result & 0x0F0F0F0F) << 4) |
           ((result >> 4) & 0x0F0F0F0F); // Swap 4-bit groups
  result = ((result & 0x00FF00FF) << 8) |
           ((result >> 8) & 0x00FF00FF); // Swap 8-bit groups
  result = ((result & 0x0000FFFF) << 16) |
           ((result >> 16) & 0x0000FFFF); // Swap 16-bit groups
  return result >> (32 - logn);           // Shift to the correct position
}
void generate_input(float2 *data, int width, int height) {
  srand(static_cast<unsigned>(time(0)));
  for (int i = 0; i < width * height; ++i) {
    data[i].x = static_cast<float>(rand()) / RAND_MAX;
    data[i].y = static_cast<float>(rand()) / RAND_MAX;
  }
}
void fft1d_cpu(float2 *data, int n) {
  int logn = static_cast<int>(log2(n));
  for (int i = 0; i < n; ++i) {
    int j = bit_reverse(i, logn);
    if (j > i) {
      float2 tmp = data[i];
      data[i] = data[j];
      data[j] = tmp;
    }
  }

  for (int s = 1; s <= logn; s++) {
    int m = 1 << s;
    float angle = -2 * PI / m;
    float2 wm = make_float2(cosf(angle), sinf(angle));

    for (int k = 0; k < n; k += m) {
      float2 w = make_float2(1, 0);
      for (int j = 0; j < m / 2; j++) {
        float2 t = complexMul(w, data[k + j + m / 2]);
        float2 u = data[k + j];
        data[k + j] = complexAdd(u, t);
        data[k + j + m / 2] = complexSub(u, t);
        w = complexMul(w, wm);
      }
    }
  }
}

void fft2d_cpu(float2 *data, int width, int height) {
  float2 *temp = new float2[width];
  for (int i = 0; i < height; ++i)
    fft1d_cpu(&data[i * width], width);

  for (int j = 0; j < width; ++j) {
    for (int i = 0; i < height; ++i)
      temp[i] = data[i * width + j];
    fft1d_cpu(temp, height);
    for (int i = 0; i < height; ++i)
      data[i * width + j] = temp[i];
  }
  delete[] temp;
}

__device__ void thread_fft1d(float2 *data, int n, int stride) {
  int logn = __ffs(n) - 1;

  // Bit reversal
  for (int i = 0; i < n; ++i) {
    int j = bit_reverse(i, logn);
    if (j > i) {
      float2 tmp = data[i * stride];
      data[i * stride] = data[j * stride];
      data[j * stride] = tmp;
    }
  }

  for (int s = 1; s <= logn; s++) {
    int m = 1 << s;
    float angle = -2 * PI / m;
    float2 wm = make_float2(cosf(angle), sinf(angle));

    for (int k = 0; k < n; k += m) {
      float2 w = make_float2(1, 0);
      for (int j = 0; j < m / 2; j++) {
        float2 t = complexMul(w, data[(k + j + m / 2) * stride]);
        float2 u = data[(k + j) * stride];
        data[(k + j) * stride] = complexAdd(u, t);
        data[(k + j + m / 2) * stride] = complexSub(u, t);
        w = complexMul(w, wm);
      }
    }
  }
}

__device__ void fft1d_shared_block(float2 *data, float2 *shared_row, int width,
                                   int stride, int block_size, int myId) {
  int tid = myId;
  int logn = __ffs(width) - 1;

  // Bit reversal
  for (int i = tid; i < width; i += block_size) {
    int j = bit_reverse(i, logn);
    if (j > i && j < width) {
      float2 temp = shared_row[i];
      shared_row[i] = shared_row[j];
      shared_row[j] = temp;
    }
  }

  __syncthreads();

  // Cooley-Tukey butterfly stages
  for (int s = 1; s <= logn; ++s) {
    int m = 1 << s;
    int half_m = m / 2;

    for (int i = tid; i < width; i += block_size) {
      int group = i / m;
      int j = i % m;
      if (j < half_m) {
        int index = group * m + j;
        float angle = -2.0f * PI * j / m;
        float2 w = make_float2(cosf(angle), sinf(angle));

        float2 u = shared_row[index];
        float2 t = complexMul(w, shared_row[(index + half_m)]);
        shared_row[index] = complexAdd(u, t);
        shared_row[(index + half_m)] = complexSub(u, t);
      }
    }

    __syncthreads();
  }
  for (int i = tid; i < width; i += block_size) {
    data[i * stride] = shared_row[i];
    // float2 test = make_float2(1.0, 1.0);
    // data[i * stride] = test;
  }
}
__global__ void thread_fft2d_col(float2 *data, int width, int height) {
  // Then do column FFTs
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < width)
    thread_fft1d(&data[col], height, width);
}

__global__ void thread_fft2d(float2 *data, int width, int height) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < height)
    thread_fft1d(&data[row * width], width, 1);
}
__global__ void block_fft2d(float2 *data, int width, int height) {
  int row = blockIdx.x;

  __shared__ float2 shared_row[WIDTH];
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    shared_row[i] = data[row * width + i];
  }

  __syncthreads();
  if (row < height)
    fft1d_shared_block(&data[row * width], shared_row, width, 1, blockDim.x,
                       threadIdx.x);
}
__global__ void block_fft2d_col(float2 *data, int width, int height) {
  int col = blockIdx.x;

  __shared__ float2 shared_row[HEIGHT];
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    shared_row[i] = data[col + width * i];
  }

  __syncthreads();
  if (col < height)
    fft1d_shared_block(&data[col], shared_row, height, width, blockDim.x,
                       threadIdx.x);
}

__global__ void warp_fft2d(float2 *data, int width, int height) {
  int warpId = threadIdx.x / 32;
  int laneId = threadIdx.x % 32;
  int row = blockIdx.x * WARPS + warpId;

  __shared__ float2 shared_row[4][WIDTH];
  for (int i = laneId; i < width; i += 32) {
    shared_row[warpId][i] = data[row * width + i];
  }
  __syncwarp();

  if (row < height) {
    fft1d_shared_block(&data[row * width], shared_row[warpId], width, 1, 32,
                       laneId);
  }
}
__global__ void warp_fft2d_col(float2 *data, int width, int height) {
  int warpId = threadIdx.x / 32;
  int laneId = threadIdx.x % 32;
  int col = blockIdx.x * WARPS + warpId;

  __shared__ float2 shared_row[4][WIDTH];
  for (int i = laneId; i < width; i += 32) {
    shared_row[warpId][i] = data[col + width * i];
  }

  if (col < height) {
    fft1d_shared_block(&data[col], shared_row[warpId], height, width, 32,
                       laneId);
  }
}
__global__ void device_fft2d(float2 *data, int width, int height, int myKernel,
                             int blocks_per_kernel) {
  int row = myKernel * blocks_per_kernel + blockIdx.x;

  __shared__ float2 shared_row[WIDTH];
  if (row < width) {
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
      shared_row[i] = data[row * width + i];
    }

    __syncthreads();
    if (row < height)
      fft1d_shared_block(&data[row * width], shared_row, width, 1, blockDim.x,
                         threadIdx.x);
  }
}
__global__ void device_fft2d_col(float2 *data, int width, int height,
                                 int myKernel, int blocks_per_kernel) {
  int col = myKernel * blocks_per_kernel + blockIdx.x;

  __shared__ float2 shared_row[WIDTH];
  if (col < width) {
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
      shared_row[i] = data[col + width * i];
    }

    __syncthreads();
    if (col < height)
      fft1d_shared_block(&data[col], shared_row, width, width, blockDim.x,
                         threadIdx.x);
  }
}

void thread_runner(float2 *data, float2 *h_cpu_result, int width, int height,
                   int size, int timed) {
  float2 *h_gpu_result = (float2 *)malloc(size);
  float2 *d_data;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  cudaMalloc(&d_data, size);
  cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

  dim3 blockDim(8);
  dim3 gridDim((height + blockDim.x - 1) / blockDim.x);

  // First launch row FFTs
  thread_fft2d<<<gridDim, blockDim>>>(d_data, width, height);
  cudaDeviceSynchronize();

  // Then launch column FFTs
  gridDim = dim3((width + blockDim.x - 1) / blockDim.x);
  thread_fft2d_col<<<gridDim, blockDim>>>(d_data, width, height);
  cudaDeviceSynchronize();

  cudaMemcpy(h_gpu_result, d_data, size, cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  float ms = 0;
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, start, end);
  if (timed) {
    printf("thread TIMING: %f\n", ms);
  }

  for (int i = 0; i < width * height; ++i) {
    float diff = fabs(h_cpu_result[i].x - h_gpu_result[i].x) +
                 fabs(h_cpu_result[i].y - h_gpu_result[i].y);
    if (diff > 2e-3) {
      printf("Mismatch at %d: CPU(%f,%f) GPU(%f,%f)\n", i, h_cpu_result[i].x,
             h_cpu_result[i].y, h_gpu_result[i].x, h_gpu_result[i].y);
    }
  }
  free(h_gpu_result);
  cudaFree(d_data);
}

void block_runner(float2 *data, float2 *h_cpu_result, int width, int height,
                  int size, int timed) {
  float2 *h_gpu_result = (float2 *)malloc(size);
  float2 *d_data;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  cudaMalloc(&d_data, size);
  cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

  dim3 blockDim(width);
  dim3 gridDim(height);

  // First launch row FFTs
  block_fft2d<<<gridDim, blockDim>>>(d_data, width, height);
  cudaDeviceSynchronize();

  // Then launch column FFTs
  // gridDim = dim3(height);
  block_fft2d_col<<<gridDim, blockDim>>>(d_data, width, height);
  cudaDeviceSynchronize();

  cudaMemcpy(h_gpu_result, d_data, size, cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  float ms = 0;
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, start, end);
  if (timed) {
    printf("block TIMING: %f\n", ms);
  }

  for (int i = 0; i < width * height; ++i) {
    float diff = fabs(h_cpu_result[i].x - h_gpu_result[i].x) +
                 fabs(h_cpu_result[i].y - h_gpu_result[i].y);
    if (diff > 1e-2) {
      printf("Mismatch at %d: CPU(%f,%f) GPU(%f,%f)\n", i, h_cpu_result[i].x,
             h_cpu_result[i].y, h_gpu_result[i].x, h_gpu_result[i].y);
    }
  }
  free(h_gpu_result);
  cudaFree(d_data);
}

void warp_runner(float2 *data, float2 *h_cpu_result, int width, int height,
                 int size, int timed) {
  float2 *h_gpu_result = (float2 *)malloc(size);
  float2 *d_data;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  cudaMalloc(&d_data, size);
  cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

  dim3 blockDim(32 * WARPS);
  dim3 gridDim(height / WARPS);

  // First launch row FFTs
  warp_fft2d<<<gridDim, blockDim>>>(d_data, width, height);
  cudaDeviceSynchronize();

  // Then launch column FFTs
  warp_fft2d_col<<<gridDim, blockDim>>>(d_data, width, height);
  cudaDeviceSynchronize();

  cudaMemcpy(h_gpu_result, d_data, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  float ms = 0;
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, start, end);
  if (timed) {
    printf("warp TIMING: %f\n", ms);
  }

  for (int i = 0; i < width * height; ++i) {
    float diff = fabs(h_cpu_result[i].x - h_gpu_result[i].x) +
                 fabs(h_cpu_result[i].y - h_gpu_result[i].y);
    if (diff > 1e-2) {
      printf("Mismatch at %d: CPU(%f,%f) GPU(%f,%f)\n", i, h_cpu_result[i].x,
             h_cpu_result[i].y, h_gpu_result[i].x, h_gpu_result[i].y);
    }
  }
  free(h_gpu_result);
  cudaFree(d_data);
}
void device_runner(float2 *data, float2 *h_cpu_result, int width, int height,
                   int size, int device_slices, int timed) {
  float2 *h_gpu_result = (float2 *)malloc(size);
  float2 *d_data;

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  cudaMalloc(&d_data, size);
  cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

  if (device_slices == 0) {
    device_slices = width;
  }
  int blocks_per_kernel = (width + device_slices - 1) / device_slices;
  dim3 blockDim(width);
  dim3 gridDim(blocks_per_kernel);
  cudaStream_t *streams =
      (cudaStream_t *)malloc(device_slices * sizeof(cudaStream_t));
  for (int i = 0; i < device_slices; i++) {
    cudaStreamCreate(&streams[i]);
  }
  for (int i = 0; i < device_slices; i++) {

    device_fft2d<<<gridDim, blockDim, 0, streams[i]>>>(d_data, width, height, i,
                                                       blocks_per_kernel);
  }
  cudaDeviceSynchronize();
  for (int i = 0; i < device_slices; i++) {

    device_fft2d_col<<<gridDim, blockDim, 0, streams[i]>>>(
        d_data, width, height, i, blocks_per_kernel);
  }
  cudaDeviceSynchronize();

  cudaMemcpy(h_gpu_result, d_data, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  float ms = 0;
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&ms, start, end);
  if (timed) {
    printf("device TIMING: %f\n", ms);
  }

  for (int i = 0; i < width * height; ++i) {
    float diff = fabs(h_cpu_result[i].x - h_gpu_result[i].x) +
                 fabs(h_cpu_result[i].y - h_gpu_result[i].y);
    if (diff > 1e-2) {
      printf("Mismatch at %d: CPU(%f,%f) GPU(%f,%f)\n", i, h_cpu_result[i].x,
             h_cpu_result[i].y, h_gpu_result[i].x, h_gpu_result[i].y);
    }
  }
  free(h_gpu_result);
  cudaFree(d_data);
}

int main(int argc, char *argv[]) {
  const int width = WIDTH;
  const int height = HEIGHT;
  int device_slices = 0;
  if (argc > 1) {
    device_slices = atoi(argv[1]);
  }
  printf("dev Slices = %d\n", device_slices);
  size_t size = width * height * sizeof(float2);

  float2 *input = (float2 *)malloc(size);
  float2 *h_cpu_result = (float2 *)malloc(size);

  generate_input(input, width, height);
  memcpy(h_cpu_result, input, size);

  struct timespec h_begin, h_end;
  // calculate vector addition on CPU
  double cpu_time = 0.0;
  clock_gettime(CLOCK_REALTIME, &h_begin);
  fft2d_cpu(h_cpu_result, width, height);
  clock_gettime(CLOCK_REALTIME, &h_end);
  cpu_time += (h_end.tv_sec - h_begin.tv_sec) * 1e3 +
              (h_end.tv_nsec - h_begin.tv_nsec) * 1e-6;

  printf("CPU TIME: %lf\n", cpu_time / 100);

  device_runner(input, h_cpu_result, width, height, size, device_slices, 0);
  warp_runner(input, h_cpu_result, width, height, size, 0);
  block_runner(input, h_cpu_result, width, height, size, 0);
  thread_runner(input, h_cpu_result, width, height, size, 0);
  device_runner(input, h_cpu_result, width, height, size, device_slices, 1);
  warp_runner(input, h_cpu_result, width, height, size, 1);
  block_runner(input, h_cpu_result, width, height, size, 1);
  thread_runner(input, h_cpu_result, width, height, size, 1);

  printf("FFT comparison complete.\n");

  free(input);
  free(h_cpu_result);
  return 0;
}
