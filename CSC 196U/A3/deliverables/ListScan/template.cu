// Graeme Knowles
// CSC196U
// A3

#include <wb.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#define BLOCK_SIZE 512 

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float *aux, int len) {
	//@@ Modify the body of this kernel to generate the scanned blocks
	//@@ Make sure to use the work efficient version of the parallel scan
	//@@ Also make sure to store the block sum to the aux array 

	// Data Load
	__shared__ float XY[2 * BLOCK_SIZE];
	const int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	const int nextI = i + blockDim.x;

	if (i < len)
	{
		XY[threadIdx.x] = input[i];
	}
	if (nextI < len)
	{
		XY[threadIdx.x + blockDim.x] = input[nextI];
	}

	// Reduction
	// XY[2*BLOCK_SIZE] is in shared memory
	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
	{
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE)
		{
			XY[index] += XY[index - stride];
		}
	}

	// Reduction Reverse
	for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) 
	{
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < 2 * BLOCK_SIZE) 
		{
			XY[index + stride] += XY[index];
		}
	}
	__syncthreads();

	// Store scanned block
	int index = 0;
	if (i < len)
	{
		index = threadIdx.x;
		output[i] = XY[index];
	}
	if (nextI < len)
	{
		index = threadIdx.x + blockDim.x;
		output[nextI] = XY[index];
	}

	// Fill aux array with sums of each block
	if (aux != NULL)
	{
		if (threadIdx.x == blockDim.x - 1)
		{
			aux[blockIdx.x] = XY[index];
		}
	}
}

__global__ void addScannedBlockSums(float *input, float *aux, int len) {
	//@@ Modify the body of this kernel to add scanned block sums to 
	//@@ all values of the scanned blocks
	if (blockIdx.x > 0)
	{
		const int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
		if (i < len)
		{
			input[i] += aux[blockIdx.x - 1];
		}
		if (i + blockDim.x < len)
		{
			input[i + blockDim.x] += aux[blockIdx.x - 1];
		}
	}
}

int main(int argc, char **argv) {
	wbArg_t args;
	float *hostInput;  // The input 1D list
	float *hostOutput; // The output 1D list
	float *deviceInput;
	float *deviceOutput;
	float *deviceAuxArray, *deviceAuxScannedArray;
	int numElements; // number of elements in the input/output list

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
	const int DATA_SIZE = numElements * sizeof(float);
	hostOutput = (float *)malloc(DATA_SIZE);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The number of input elements in the input is ", numElements);

	wbTime_start(GPU, "Allocating device memory.");
	//@@ Allocate device memory
	//you can assume that aux array size would not need to be more than BLOCK_SIZE*2 (i.e., 1024)
	const int NUM_BLOCKS = ((numElements - 1) / BLOCK_SIZE) + 1;
	cudaMalloc((void **)&deviceInput, DATA_SIZE);
	cudaMalloc((void **)&deviceOutput, DATA_SIZE);
	cudaMalloc((void **)&deviceAuxArray, BLOCK_SIZE * 2 * sizeof(float));
	cudaMalloc((void **)&deviceAuxScannedArray, NUM_BLOCKS * sizeof(float));
	wbTime_stop(GPU, "Allocating device memory.");

	wbTime_start(GPU, "Clearing output device memory.");
	wbCheck(cudaMemset(deviceOutput, 0, DATA_SIZE));
	wbTime_stop(GPU, "Clearing output device memory.");

	wbTime_start(GPU, "Copying input host memory to device.");
	//@@ Copy input host memory to device	
	cudaMemcpy(deviceInput, hostInput, DATA_SIZE, cudaMemcpyHostToDevice);
	wbTime_stop(GPU, "Copying input host memory to device.");

	//@@ Initialize the grid and block dimensions here
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(NUM_BLOCKS, 1, 1);

	dim3 dimBlockAux(NUM_BLOCKS, 1, 1);
	dim3 dimGridAux(1, 1, 1);

	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Modify this to complete the functionality of the scan
	//@@ on the deivce
	//@@ You need to launch scan kernel twice: 1) for generating scanned blocks
	//@@ (hint: pass deviceAuxArray to the aux parameter)
	scan <<<dimGrid, dimBlock>>> (deviceInput, deviceOutput, deviceAuxArray, numElements);

	//@@ and 2) for generating scanned aux array that has the scanned block sums. 
	//@@ (hint: pass NULL to the aux parameter)
	scan <<<dimGridAux, dimBlockAux>>> (deviceAuxArray, deviceAuxScannedArray, NULL, NUM_BLOCKS);

	//@@ Then you should call addScannedBlockSums kernel.
	addScannedBlockSums <<<dimGrid, dimBlock>>>(deviceOutput, deviceAuxScannedArray, numElements);

	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output device memory to host");
	//@@ Copy results from device to host
	cudaMemcpy(hostOutput, deviceOutput, DATA_SIZE, cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying output device memory to host");

	wbTime_start(GPU, "Freeing device memory");
	//@@ Deallocate device memory
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	cudaFree(deviceAuxArray);
	cudaFree(deviceAuxScannedArray);
	wbTime_stop(GPU, "Freeing device memory");

	wbSolution(args, hostOutput, numElements);

	free(hostInput);
	free(hostOutput);

	return 0;
}
