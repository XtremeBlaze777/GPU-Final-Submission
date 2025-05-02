#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

//histogram calculation on CPU
void serial_histogram(int *A, int N, int* result) {
    for (int i=0; i<N; i++) {
        result[A[i]]++ ; 
    }
}

// Histogram on GPU
__global__ void naive_parallel_histogram(int *A, int N, int *result) {
    int id = blockDim.x * blockIdx.x + threadIdx.x ;
    
    if(id < N)  {
        //update partial histogram in global memory
        atomicAdd(&result[A[id]], 1);
    }
    
}

// Histogram on GPU using shared memory
__global__ void block_histogram(int *A, int N, int *result) {
    int id = blockDim.x * blockIdx.x + threadIdx.x ;
    int tid = threadIdx.x;
    __shared__ int tileSh[256];

    if(tid<256) {       
        //initialize shared memory with zero 
        tileSh[tid] = 0;
    }
    __syncthreads();
    if(id < N)  {
        //update partial histogram in shared memory
        atomicAdd(&tileSh[A[id]], 1);
    }
    __syncthreads();
    if(tid<256) {
        //update global memory with partial results in shared memory
        atomicAdd(&result[tid], tileSh[tid]);
    }
}

__global__ void warp_histogram(int *A, int N, int *result, int num_blocks) {
    int id = blockDim.x * blockIdx.x + threadIdx.x ;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32; 

    __shared__ int warpHist[32][256]; // first one is warps per block

    for (int i=lane; i<256; i+=32) {
        warpHist[warp_id][i] = 0;
    }
    __syncwarp();

    if (id < N)  {
        atomicAdd(&warpHist[warp_id][A[id]], 1);
    }
    __syncwarp();

    // One thread per bin aggregates per-warp results
    for (int i=lane; i<256; i+=32) {
        atomicAdd(&result[i], warpHist[warp_id][i]);
    }
}

__global__ void device_histogram(int *A, int N, int *result) {
    cg::grid_group grid = cg::this_grid();
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ int tileSh[256];
    if (tid < 256) {
        tileSh[tid] = 0;
    }

    __syncthreads();

    // Guard access with id < N
    if (id < N) {
        atomicAdd(&tileSh[A[id]], 1);
    }

    // === DEVICE-WIDE SYNC ===
    // Every thread must hit this line, even if id >= N
    grid.sync();

    // Every block contributes to global histogram
    if (tid < 256) {
        atomicAdd(&result[tid], tileSh[tid]);
    }
}

void compare_results(int * result, int* out) {
    for(int i=0; i<256; i++) {
        if(result[i] != out[i]) {
            fprintf(stderr, "incorrect logic on GPU!\n");
            return;   
        }
    }
    // printf("Correctnes check done!\n");
}



int main(int argc, char* argv[]) {
    cudaFree(0);  //initialize CUDA

    int N = 1000000;
    if(argc == 2) {
        N = atoi(argv[1]);
    }

    size_t bytes = sizeof(int) * N;
    int *A = (int *) malloc(bytes);
    int *result = (int *) calloc(256, sizeof(int));  // CPU result
    int *out = (int *) calloc(256, sizeof(int));     // GPU output

    for (int i = 0; i < N; i++) {
        A[i] = rand() % 256;
    }

    // Time CPU histogram
    struct timespec begin, end;
    clock_gettime(CLOCK_REALTIME, &begin);
    serial_histogram(A, N, result);
    clock_gettime(CLOCK_REALTIME, &end);
    double elapsed_msec = (end.tv_sec - begin.tv_sec) * 1e3 + (end.tv_nsec - begin.tv_nsec) * 1e-6;

    // Allocate device memory
    int *A_d, *out_d;
    cudaMalloc(&A_d, bytes);
    cudaMalloc(&out_d, 256 * sizeof(int));
    cudaMemcpy(A_d, A, bytes, cudaMemcpyHostToDevice);

    const int blk_sz = 32 * 32;
    dim3 blk_dim(blk_sz);
    dim3 grid_dim((N + blk_sz - 1) / blk_sz);

    cudaEvent_t begin_d, end_d;
    cudaEventCreate(&begin_d);
    cudaEventCreate(&end_d);

    // === Naive GPU Histogram Timing ===
    // float total_time_naive = 0.0f;
    // for (int i = 0; i < 1; ++i) {
    //     cudaMemset(out_d, 0, 256 * sizeof(int));
    //     cudaEventRecord(begin_d, 0);
    //     naive_parallel_histogram<<<grid_dim, blk_dim>>>(A_d, N, out_d);
    //     cudaEventRecord(end_d, 0);
    //     cudaEventSynchronize(end_d);

	// 	cudaDeviceSynchronize();

    //     float time_ms;
    //     cudaEventElapsedTime(&time_ms, begin_d, end_d);
    //     total_time_naive += time_ms;
    // }
    // cudaMemcpy(out, out_d, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("thread histogram (ms): %.6f\n", total_time_naive / 1.0f);
	// fflush(stdout);
    // compare_results(result, out);

    // === Shared Memory GPU Histogram Timing ===
    float total_time_shared = 0.0f;
    for (int i = 0; i < 1; ++i) {
        cudaMemset(out_d, 0, 256 * sizeof(int));
        cudaEventRecord(begin_d, 0);
        block_histogram<<<grid_dim, blk_dim>>>(A_d, N, out_d);
        cudaEventRecord(end_d, 0);
        cudaEventSynchronize(end_d);

		cudaDeviceSynchronize();

        float time_ms;
        cudaEventElapsedTime(&time_ms, begin_d, end_d);
        total_time_shared += time_ms;
    }
    cudaMemcpy(out, out_d, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("block histogram (ms): %.6f\n", total_time_shared / 1.0f);
	fflush(stdout);
    compare_results(result, out);


	// === Warp-level GPU Histogram Timing ===
	float total_time_warp = 0.0f;
	for (int i = 0; i < 100; ++i) {
		cudaMemset(out_d, 0, 256 * sizeof(int));
		cudaEventRecord(begin_d, 0);
		warp_histogram<<<grid_dim, blk_dim>>>(A_d, N, out_d, grid_dim.x);
		cudaEventRecord(end_d, 0);
		cudaEventSynchronize(end_d);
		
		cudaDeviceSynchronize();

		float time_ms;
		cudaEventElapsedTime(&time_ms, begin_d, end_d);
		total_time_warp += time_ms;
	}
	cudaMemcpy(out, out_d, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	printf("warp histogram (ms): %.6f\n", total_time_warp / 100.0f);
	fflush(stdout);
	compare_results(result, out);


	// === Cooperative Groups GPU Histogram Timing ===
	float total_time_device = 0.0f;
	for (int i = 0; i < 100; i++) {
		cudaMemset(out_d, 0, 256 * sizeof(int));
		
		cudaEventRecord(begin_d, 0);
		void* kernelArgs[] = { &A_d, &N, &out_d };
		cudaLaunchCooperativeKernel((void*)device_histogram, grid_dim, blk_dim, kernelArgs);
		cudaEventRecord(end_d, 0);
		cudaEventSynchronize(end_d);
	
		float iter_time = 0.0f;
		cudaEventElapsedTime(&iter_time, begin_d, end_d);
		total_time_device += iter_time;
	}
	cudaMemcpy(out, out_d, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	printf("device histogram (ms): %.6f\n", total_time_device / 100.0f);
	fflush(stdout);
	compare_results(result, out);


    cudaFree(A_d);
    cudaFree(out_d);
    free(A);
    free(result);
    free(out);
}