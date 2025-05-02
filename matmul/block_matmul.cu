#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<sys/time.h>
#include<math.h> 
#include <cassert>
using namespace std;
#define TILE_WIDTH 16

// reference: https://github.com/debowin/cuda-tiled-matrix-multiplication/blob/master/matrixmul.cu
// reference: https://siboehm.com/articles/22/CUDA-MMM

void verify_result(unsigned long * result, unsigned long* C, int width) {
	for(int i=0; i<width*width; i++) {
		//printf("%d\t%d\t%d\n", i, C[i], result[i]);
		if(result[i] != C[i]) {
			printf("Not equal\n");
			return;
		}
	}
	printf("Verification done. Result on GPU and CPU same.\n");
}
void cpu_mm (int * A, int * B, unsigned long *result, int width) {
	struct timeval begin, end;
        gettimeofday(&begin, NULL);
        //struct timespec begin, end;
		//clock_gettime(CLOCK_REALTIME, &begin);
        for(int i=0; i<width; i++) {
                //int temp=0;
		for(int j=0; j<width; j++){
                        int temp=0;
			for(int k=0; k<width; k++) {
                                temp += A[i*width+k] * B[k*width+j];
                        }
			result[i*width+j] = temp;
                }
        }
        gettimeofday(&end, NULL);
        //clock_gettime(CLOCK_REALTIME, &end);
		long seconds = end.tv_sec - begin.tv_sec;
    	long useconds = end.tv_usec - begin.tv_usec;
	double elapsed = seconds*1e3 + useconds*1e-3; // nanoseconds*1e-9;
    	printf("Time measured on CPU: %.6f ms.\n", elapsed);

}


__global__ void kernel_gpu_mm(int * A , int * B ,  unsigned long * C , int width) {
	int row = blockIdx . y * blockDim . y + threadIdx . y ;
        int col = blockIdx . x * blockDim . x + threadIdx . x ;
	if ( row < width && col < width ) {
                unsigned long temp = 0;
                for(int i=0; i<width; i++) {
                	temp += A [ row * width + i ] *B [ i * width + col ];
                }
		C [ row * width + col ] = temp;
        }
}

void gpu_mm(int * A, int * B,  int width, unsigned long * result) {
	unsigned long *C = (unsigned long *) malloc ( (width*width) * sizeof (unsigned long) );
	dim3 dimGrid (( width + 15) / 16 , ( width + 15) / 16 , 1);
	dim3 dimBlock (16 , 16 , 1);
	// Allocate device memory for matrices ’A ’, ’B ’, and ’C ’
        // TODO : Allocate device memory
        int *gpuA, *gpuB;
        unsigned long *gpuC;
        cudaMalloc( &gpuA, width * width* sizeof(int) );
        cudaMalloc( &gpuB, width * width* sizeof(int) );
        cudaMalloc( &gpuC, width * width* sizeof(unsigned long) );

        // First kernel call to setup the device
        cudaMemcpy(gpuA, A, width * width* sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuB, B, width * width* sizeof(int), cudaMemcpyHostToDevice);
        kernel_gpu_mm<<<dimGrid, dimBlock>>>(gpuA, gpuB, gpuC, width);
        cudaDeviceSynchronize();
        cudaMemset(gpuC, 0, width * width* sizeof(unsigned long));
        cudaDeviceSynchronize();

	cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliseconds = 0;
        cudaEventRecord(start,0);
        // TODO : Copy matrices from host to device
        cudaMemcpy(gpuA, A, width * width* sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuB, B, width * width* sizeof(int), cudaMemcpyHostToDevice);

	//printf("calling kernel_ghpu_mm\n");
	kernel_gpu_mm <<<dimGrid, dimBlock>>>(gpuA, gpuB, gpuC, width);
	
	//cudaDeviceSynchronize();
	//printf("done kernel_gpu_mm\n");
        // Copy the result matrix ’C ’ from device to host
        // TODO : Copy matrix ’C ’ from device to host
        cudaMemcpy(C, gpuC, width * width* sizeof(unsigned long), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time taken by naive KERNEL to execute is: %.6f ms\n", milliseconds);
	cudaFree(gpuA);
        cudaFree(gpuB);
        cudaFree(gpuC);
	verify_result(C, C, width);
	free(C);
}

__global__ void kernel_gpu_shm_mm(int * A , int * B , unsigned long* C , int width) {
    __shared__ int tileAs[TILE_WIDTH][TILE_WIDTH];
    __shared__ int tileBs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;

    // target element coordinates
    int row = by * TILE_WIDTH + ty;
    int column = bx * TILE_WIDTH + tx;

    unsigned long pValue = 0;
    for(int i=0; i<ceilf(width/(float)TILE_WIDTH); i++) {
            // move the tiles and update shared memory value for new tile positions
            if(row < width && (i*TILE_WIDTH + tx)<width)
                    tileAs[ty][tx] = A[row*width + i*TILE_WIDTH + tx];
            else
                    tileAs[ty][tx] = 0;
            if(column < width && (i*TILE_WIDTH + ty)<width)
                    tileBs[ty][tx] = B[(i*TILE_WIDTH + ty)*width + column];
            else
                    tileBs[ty][tx] = 0;

            // after the entire tile's values are available, proceed
            __syncthreads();

            for(int j=0; j<TILE_WIDTH; j++)
                    pValue += tileAs[ty][j] * tileBs[j][tx];
            // after the entire tile's values have been used, proceed
            __syncthreads();
    }
    // boundary check
    if(row < width && column < width)
            C[row*width+column] = pValue;	
}

void gpu_shm_mm (int * A, int * B, int width, unsigned long* result) {
	dim3 dimGrid (( width + (TILE_WIDTH-1)) / TILE_WIDTH , ( width + (TILE_WIDTH-1)) / TILE_WIDTH , 1);
        dim3 dimBlock (TILE_WIDTH, TILE_WIDTH, 1);
	unsigned long * C = (unsigned long *) malloc ( (width*width) * sizeof (unsigned long) );
	int *gpuA, *gpuB;
        unsigned long *gpuC;
        cudaMalloc( &gpuA, width * width* sizeof(int) );
        cudaMalloc( &gpuB, width * width* sizeof(int) );
        cudaMalloc( &gpuC, width * width* sizeof(unsigned long) );

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliseconds = 0;
        cudaEventRecord(start,0);
        // TODO : Copy matrices from host to device
        cudaMemcpy(gpuA, A, width * width* sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuB, B, width * width* sizeof(int), cudaMemcpyHostToDevice);

        kernel_gpu_shm_mm <<<dimGrid, dimBlock>>>(gpuA, gpuB, gpuC, width);

        //cudaDeviceSynchronize();
        // Copy the result matrix ’C ’ from device to host
        // TODO : Copy matrix ’C ’ from device to host
        cudaMemcpy(C, gpuC, width * width* sizeof(unsigned long), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time taken by shared memory KERNEL to execute is: %.6f ms\n", milliseconds);
	cudaFree(gpuA);
        cudaFree(gpuB);
        cudaFree(gpuC);
	verify_result(C, C, width);
	free(C);
}


int main(int argc, char *argv[]) {
	unsigned long width = 2048;
	if(argc == 2 ) {
                width = strtoul(argv[1], NULL, 10);
        }
	printf("width : %d\n", width);
	int *A = (int*) malloc ( (width*width) * sizeof (int) );
        int *B = (int*) malloc ( (width*width) * sizeof (int) );
       
	//initializing matrices, A and B
	for(int j=0; j<width*width; j++) {
        	A[j] = rand()%1000;
        	B[j] = rand()%1000;
	}
	unsigned long *result = (unsigned long *) malloc ( (width*width) * sizeof (unsigned long) );
	// calculate matrix multiplication on CPU
	// cpu_mm (A, B, result, width);
        // printf("CPU calculation done\n");
	// matrix multiplication on GPU

	gpu_mm (A, B,  width, result);

	// GPU matrix mulitplication using shared memory

	gpu_shm_mm(A, B,  width, result);

	free(result);
        free(A);
        free(B);
	return 0;

}


