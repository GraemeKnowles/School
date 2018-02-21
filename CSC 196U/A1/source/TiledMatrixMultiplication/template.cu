#include <wb.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TILE_SIZE 32

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

struct Tile {
	int width;
	int height;
	int stride;
	float* elements;
};

__device__ float getElement(Tile * const tile, int row, int column)
{
	return tile->elements[row * tile->stride + column];
}

__device__ Tile getTile(float * const matrix, int row, int stride, int column)
{
	Tile t;
	t.width = t.height = TILE_SIZE;
	t.stride = stride;
	t.elements = &matrix[row * stride * TILE_SIZE + column * TILE_SIZE];
	return t;
}

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) 
{
  //@@ Insert code to implement tiled matrix multiplication here
  //@@ You have to use shared memory to write this kernel

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("blockX=%i blockY=%i\n", blockIdx.x, blockIdx.y);

	if ((row < numCRows) && (col < numCColumns))
	{
		__shared__ float aTile[TILE_SIZE][TILE_SIZE];
		__shared__ float bTile[TILE_SIZE][TILE_SIZE];

		int sharedRow = threadIdx.y;
		int sharedCol = threadIdx.x;

		//printf("threadX=%i threadY=%i\n", threadIdx.x, threadIdx.y);
		
		const int numTiles = numAColumns / TILE_SIZE;

		float sum = 0;

		for (int i = 0; i < numTiles; ++i)
		{
			Tile tA = getTile(A, blockIdx.y, numAColumns, i);
			Tile tB = getTile(B, i, numBColumns, blockIdx.x);

			if (threadIdx.x == 0 && threadIdx.y == 0)
			{
				printf("TileA %i col=%i row=%i\n", i, blockIdx.y * TILE_SIZE, i * TILE_SIZE);
				printf("TileB %i col=%i row=%i\n", i, i * TILE_SIZE, blockIdx.x * TILE_SIZE);
			}
			
			aTile[sharedRow][sharedCol] = getElement(&tA, sharedRow, sharedCol);
			bTile[sharedRow][sharedCol] = getElement(&tB, sharedRow, sharedCol);
			__syncthreads();// Sync Reads

			for (int j = 0; j < TILE_SIZE; ++j)
			{
				const int index = i * TILE_SIZE + j;
				if (index < numAColumns && index < numBRows)
				{
					sum += aTile[sharedRow][j] * bTile[j][sharedCol];
				}
			}
			__syncthreads();// Sync Writes
		}

		C[row * numCColumns + col] = sum;
	}
}

int main(int argc, char **argv) {
	wbArg_t args;
	float *hostA; // The A matrix
	float *hostB; // The B matrix
	float *hostC; // The output C matrix
	float *deviceA;
	float *deviceB;
	float *deviceC;
	int numARows;    // number of rows in the matrix A
	int numAColumns; // number of columns in the matrix A
	int numBRows;    // number of rows in the matrix B
	int numBColumns; // number of columns in the matrix B
	int numCRows;    // number of rows in the matrix C (you have to set this)
	int numCColumns; // number of columns in the matrix C (you have to set
					 // this)
	hostC = NULL;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
		&numAColumns);
	const int aSize = numARows * numAColumns * sizeof(float);
	hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
		&numBColumns);
	const int bSize = numBRows * numBColumns * sizeof(float);

	//@@ Set numCRows and numCColumns
	numCRows = numARows;
	numCColumns = numBColumns;
	const int cElements = numCRows * numCColumns;
	const int cSize = cElements * sizeof(float);

	//@@ Allocate the hostC matrix
	wbTime_stop(Generic, "Importing data and creating memory on host");
	hostC = new float[cElements];

	wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
	wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
	wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

	wbTime_start(GPU, "Allocating GPU memory.");
	//@@ Allocate GPU memory here
	cudaMalloc((void **)&deviceA, aSize);
	cudaMalloc((void **)&deviceB, bSize);
	cudaMalloc((void **)&deviceC, cSize);
	wbTime_stop(GPU, "Allocating GPU memory.");
	wbCheck(cudaGetLastError());

	wbTime_start(GPU, "Copying input memory to the GPU.");
	//@@ Copy memory to the GPU here
	cudaMemcpy(deviceA, hostA, aSize, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, bSize, cudaMemcpyHostToDevice);
	wbTime_stop(GPU, "Copying input memory to the GPU.");
	wbCheck(cudaGetLastError());

	//@@ Initialize the grid and block dimensions here
	const int gridSize = TILE_SIZE;
	const int gridY = (numCColumns - 1) / gridSize + 1;
	const int gridX = (numCRows - 1) / gridSize + 1;
	wbLog(TRACE, "The grid dimensions are ", gridX, " x ", gridY);
	wbLog(TRACE, "The block dimensions are ", gridSize, " x ", gridSize);
	dim3 dimGrid(gridY, gridX, 1);
	dim3 dimBlock(gridSize, gridSize, 1);

	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Launch the GPU Kernel here
	matrixMultiplyShared << <dimGrid, dimBlock >> > (
		deviceA, deviceB, deviceC,
		numARows, numAColumns,
		numBRows, numBColumns,
		numCRows, numCColumns
		);

	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");
	wbCheck(cudaGetLastError());

	wbTime_start(Copy, "Copying output memory to the CPU");
	//@@ Copy the GPU memory back to the CPU here
	cudaMemcpy(hostC, deviceC, cSize, cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying output memory to the CPU");
	wbCheck(cudaGetLastError());

	wbTime_start(GPU, "Freeing GPU Memory");
	//@@ Free the GPU memory here
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
	wbTime_stop(GPU, "Freeing GPU Memory");
	wbCheck(cudaGetLastError());

	wbSolution(args, hostC, numCRows, numCColumns);

	free(hostA);
	free(hostB);
	free(hostC);

	return 0;
}
