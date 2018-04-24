// Graeme Knowles
// CSC196U
// A3

#include <wb.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#define NUM_BINS 4096
#define BLOCK_SIZE 512 
#define MAX_BIN 127

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
	bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
			file, line);
		if (abort)
			exit(code);
	}
}

__global__ void histogram(unsigned int *input, unsigned int *bins,
	unsigned int num_elements,
	unsigned int num_bins)
{
	//@@ Write the kernel that computes the histogram
	//@@ Make sure to use the privitization technique

	// Block private Histogram
	__shared__ unsigned int local_histo[NUM_BINS];

	// Initialize local histogram to 0
	const int binStride = blockDim.x;
	for (int binInc = threadIdx.x; binInc < NUM_BINS; binInc += binStride)
	{
		local_histo[binInc] = 0;
	}
	
	__syncthreads();

	// Build Histogram
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (int inc = i, stride = blockDim.x * gridDim.x; 
		inc < num_elements; inc += stride)
	{
		atomicAdd(&(local_histo[input[inc]]), 1);
	}

	__syncthreads();

	// Add local histogram to global histogram
	for (int binInc = threadIdx.x; binInc < NUM_BINS; binInc += binStride)
	{
		atomicAdd(&(bins[binInc]), local_histo[binInc]);
	}
}

__global__ void saturate(unsigned int *bins, unsigned int num_bins) {
	//@@ Write the kernel that applies saturtion to counters (i.e., if the bin value is more than 127, make it equal to 127)
	const int binStride = blockDim.x;
	for (int binInc = threadIdx.x; binInc < NUM_BINS; binInc += binStride)
	{
		if (bins[binInc] > MAX_BIN)
		{
			bins[binInc] = MAX_BIN;
		}
	}
}

int main(int argc, char *argv[]) {
	wbArg_t args;
	int inputLength;
	const unsigned int BIN_SIZE = NUM_BINS * sizeof(unsigned int);
	unsigned int *hostInput;
	unsigned int *hostBins;
	unsigned int *deviceInput;
	unsigned int *deviceBins;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
		&inputLength, "Integer");
	const unsigned int INPUT_SIZE = inputLength * sizeof(unsigned int);
	hostBins = (unsigned int *)malloc(BIN_SIZE);

	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);
	wbLog(TRACE, "The number of bins is ", NUM_BINS);

	wbTime_start(GPU, "Allocating device memory");
	//@@ Allocate device memory here
	cudaMalloc((void **)&deviceInput, INPUT_SIZE);
	cudaMalloc((void **)&deviceBins, BIN_SIZE);
	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Allocating device memory");

	wbTime_start(GPU, "Copying input host memory to device");
	//@@ Copy input host memory to device
	cudaMemcpy(deviceInput, hostInput, INPUT_SIZE, cudaMemcpyHostToDevice);
	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Copying input host memory to device");

	wbTime_start(GPU, "Clearing the bins on device");
	//@@ zero out the deviceBins using cudaMemset() 
	CUDA_CHECK(cudaMemset(deviceBins, 0, BIN_SIZE));
	wbTime_stop(GPU, "Clearing the bins on device");

	//@@ Initialize the grid and block dimensions here
	const int NUM_BLOCKS = ((NUM_BINS - 1) / BLOCK_SIZE) + 1;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(NUM_BLOCKS, 1, 1);

	wbLog(TRACE, "Launching kernel");
	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Invoke kernels: first call histogram kernel and then call saturate kernel
	histogram << <dimGrid, dimBlock >> > (deviceInput, deviceBins, inputLength, NUM_BINS);
	saturate << <dimGrid, dimBlock >> > (deviceBins, NUM_BINS);
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output device memory to host");
	//@@ Copy output device memory to host
	cudaMemcpy(hostBins, deviceBins, BIN_SIZE, cudaMemcpyDeviceToHost);
	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(Copy, "Copying output device memory to host");

	wbTime_start(GPU, "Freeing device memory");
	//@@ Free the device memory here
	cudaFree(deviceInput);
	cudaFree(deviceBins);
	wbTime_stop(GPU, "Freeing device memory");

	wbSolution(args, hostBins, NUM_BINS);

	free(hostBins);
	free(hostInput);
	return 0;
}
