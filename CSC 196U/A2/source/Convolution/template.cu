#include <wb.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#define MASK_WIDTH 5
#define O_TILE_WIDTH 16
#define BLOCK_WIDTH O_TILE_WIDTH + (MASK_WIDTH-1)
#define NUM_CHANNELS 3
#define clamp(x) (min(max((x), 0.0), 1.0))

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ INSERT CODE HERE 
//implement the tiled 2D convolution kernel with adjustments for channels
//use shared memory to reduce the number of global accesses, handle the boundary conditions when loading input list elements into the shared memory
//clamp your output values

__global__ void convolution(
	float const * const __restrict__ mask,
	float const * const inputImage, float * const outputImage,
	const int imageWidth, const int imageHeight)
{
	// Each block loads a tile
	__shared__ float inputTile[BLOCK_WIDTH][BLOCK_WIDTH];

	static const int H_MASK = MASK_WIDTH / 2;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int tileRow = ty - H_MASK;
	const int tileCol = tx - H_MASK;
	const int row_o = blockIdx.y * O_TILE_WIDTH + ty;
	const int col_o = blockIdx.x * O_TILE_WIDTH + tx;
	const int row_i = row_o - H_MASK;
	const int col_i = col_o - H_MASK;

	const int channel = blockIdx.z;

	// If the current thread corresponds to a valid data element
	bool validElement = (row_i >= 0 && row_i < imageHeight) && (col_i >= 0 && col_i < imageWidth);

	// Load tile, all threads participate
	if (validElement)
	{
		inputTile[ty][tx] = inputImage[(row_i * imageWidth + col_i) * gridDim.z + channel];
	}
	else // if invalid, load 0
	{
		inputTile[ty][tx] = 0.0f;
	}
	
	__syncthreads();// Sync Reads

	if (validElement && tileRow >= 0 && tileRow < O_TILE_WIDTH && tileCol >= 0 && tileCol < O_TILE_WIDTH)
	{
		float pixVal = 0;
		// iterate through mask
		for (int mRow = 0; mRow < MASK_WIDTH; ++mRow)
		{
			int curRow = tileRow + mRow;
			for (int mCol = 0; mCol < MASK_WIDTH; ++mCol) 
			{
				int curCol = tileCol + mCol;
				pixVal += inputTile[curRow][curCol] * mask[mRow * MASK_WIDTH + mCol];
			}
		}

		// Write convoluted pixel data
		outputImage[(row_i * imageWidth + col_i) * gridDim.z + channel] = clamp(pixVal);
	}
	__syncthreads();// Sync writes
}

int main(int argc, char *argv[]) {
	wbArg_t arg;
	int maskRows;
	int maskColumns;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	char *inputMaskFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *hostMaskData;
	float *deviceInputImageData;
	float *deviceOutputImageData;
	float *deviceMaskData;

	arg = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(arg, 0);
	inputMaskFile = wbArg_getInputFile(arg, 1);

	inputImage = wbImport(inputImageFile);
	hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

	assert(maskRows == MASK_WIDTH);    /* mask height is fixed to 5 */
	assert(maskColumns == MASK_WIDTH); /* mask width is fixed to 5 */
	static const int MASK_SIZE = maskRows * maskColumns * sizeof(float);

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	//@@ INSERT CODE HERE
	const int IMAGE_SIZE = imageWidth * imageHeight * imageChannels * sizeof(float);
	cudaMalloc((void **)&deviceInputImageData, IMAGE_SIZE);
	cudaMalloc((void **)&deviceOutputImageData, IMAGE_SIZE);
	cudaMalloc((void **)&deviceMaskData, MASK_SIZE);
	//allocate device memory
	wbTime_stop(GPU, "Doing GPU memory allocation");
	wbCheck(cudaGetLastError());

	wbTime_start(Copy, "Copying data to the GPU");
	//@@ INSERT CODE HERE
	//copy host memory to device
	cudaMemcpy(deviceInputImageData, hostInputImageData, IMAGE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData, hostMaskData, MASK_SIZE, cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");
	wbCheck(cudaGetLastError());

	wbTime_start(Compute, "Doing the computation on the GPU");
	//@@ INSERT CODE HERE
	//initialize thread block and kernel grid dimensions
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 dimGrid((wbImage_getWidth(inputImage) - 1) / O_TILE_WIDTH + 1, (wbImage_getHeight(inputImage) - 1) / O_TILE_WIDTH + 1, imageChannels);

	//invoke CUDA kernel
	convolution <<<dimGrid, dimBlock>>> (deviceMaskData, deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);

	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Doing the computation on the GPU");
	wbCheck(cudaGetLastError());

	wbTime_start(Copy, "Copying data from the GPU");
	//@@ INSERT CODE HERE
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, IMAGE_SIZE, cudaMemcpyDeviceToHost);
	//copy results from device to host	
	wbTime_stop(Copy, "Copying data from the GPU");
	wbCheck(cudaGetLastError());

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
	
	wbSolution(arg, outputImage);

	//@@ INSERT CODE HERE
	//deallocate device memory	
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceMaskData);
	free(hostMaskData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
