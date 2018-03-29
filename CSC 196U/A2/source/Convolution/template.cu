// Graeme Knowles
// CSC 196U
// Assignment 2

#include <wb.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#define MASK_WIDTH 5
#define O_TILE_WIDTH 16
#define BLOCK_WIDTH O_TILE_WIDTH + (MASK_WIDTH-1)
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

	// Half the mask size - integer division
	const int HALF_MASK = MASK_WIDTH / 2;
	// Pixel indices
	const int TX = threadIdx.x;
	const int TY = threadIdx.y;
	const int TILE_ROW = TY - HALF_MASK;
	const int TILE_COL = TX - HALF_MASK;
	const int ROW_O = blockIdx.y * O_TILE_WIDTH + TY;
	const int COL_O = blockIdx.x * O_TILE_WIDTH + TX;
	const int ROW_I = ROW_O - HALF_MASK;
	const int COL_I = COL_O - HALF_MASK;
	// Channel indices
	const int NUM_CHANNELS = gridDim.z;
	const int CUR_CHANNEL = blockIdx.z;

	// If the current thread corresponds to a valid data element
	const bool ELEM_VAL = (ROW_I >= 0 && ROW_I < imageHeight) && (COL_I >= 0 && COL_I < imageWidth);

	// Load tile, all threads participate
	if (ELEM_VAL)
	{
		inputTile[TY][TX] = inputImage[(ROW_I * imageWidth + COL_I) * NUM_CHANNELS + CUR_CHANNEL];
	}
	else // if invalid, load 0
	{
		inputTile[TY][TX] = 0.0f;
	}

	__syncthreads();// Sync Reads

	if (ELEM_VAL && TILE_ROW >= 0 && TILE_ROW < O_TILE_WIDTH && TILE_COL >= 0 && TILE_COL < O_TILE_WIDTH)
	{
		float pixVal = 0;
		// iterate through mask
		for (int mRow = 0; mRow < MASK_WIDTH; ++mRow)
		{
			const int CUR_ROW = TILE_ROW + mRow;
			for (int mCol = 0; mCol < MASK_WIDTH; ++mCol)
			{
				const int CUR_COL = TILE_COL + mCol;
				pixVal += inputTile[CUR_ROW][CUR_COL] * mask[mRow * MASK_WIDTH + mCol];
			}
		}

		// Write convoluted pixel data
		outputImage[(ROW_I * imageWidth + COL_I) * NUM_CHANNELS + CUR_CHANNEL] = clamp(pixVal);
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
	//allocate device memory
	const int IMAGE_SIZE = imageWidth * imageHeight * imageChannels * sizeof(float);
	cudaMalloc((void **)&deviceInputImageData, IMAGE_SIZE);
	cudaMalloc((void **)&deviceOutputImageData, IMAGE_SIZE);
	cudaMalloc((void **)&deviceMaskData, MASK_SIZE);
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
	// Since the convolution is done on each channel separately,
	// can make use of the 3rd grid dimension to handle each channel
	dim3 dimGrid(
		(wbImage_getWidth(inputImage) - 1) / O_TILE_WIDTH + 1,
		(wbImage_getHeight(inputImage) - 1) / O_TILE_WIDTH + 1,
		imageChannels);
	//invoke CUDA kernel
	convolution << <dimGrid, dimBlock >> > (
		deviceMaskData,
		deviceInputImageData, deviceOutputImageData,
		imageWidth, imageHeight);
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Doing the computation on the GPU");
	wbCheck(cudaGetLastError());

	wbTime_start(Copy, "Copying data from the GPU");
	//@@ INSERT CODE HERE
	//copy results from device to host	
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, IMAGE_SIZE, cudaMemcpyDeviceToHost);
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
