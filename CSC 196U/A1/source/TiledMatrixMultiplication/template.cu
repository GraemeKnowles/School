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

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
	int numARows, int numAColumns,
	int numBRows, int numBColumns,
	int numCRows, int numCColumns)
{
	//@@ Insert code to implement tiled matrix multiplication here
	//@@ You have to use shared memory to write this kernel

	// Check to make sure the current thread index is a valid
	// element in the c matrix
	const int C_ROW = blockIdx.y * blockDim.y + threadIdx.y;
	const int C_COL = blockIdx.x * blockDim.x + threadIdx.x;
	const bool VALID_C_IND = C_ROW < numCRows && C_COL < numCColumns;

	// Rename for clearer code
	const int TILE_X = threadIdx.x, TILE_Y = threadIdx.y;

	// Shared memory allocation
	// tile[x][y] x = row, y = column
	__shared__ float aTile[TILE_SIZE][TILE_SIZE];
	__shared__ float bTile[TILE_SIZE][TILE_SIZE];

	float partialDotProduct = 0;
	const int numTiles = (numAColumns - 1) / TILE_SIZE + 1;
	for (int i = 0; i < numTiles; ++i)
	{
		// The row that is the top of the A tile
		const int A_TILE_ROW = blockIdx.y * TILE_SIZE;
		// The column that is the left of the B tile
		const int B_TILE_COL = blockIdx.x * TILE_SIZE;
		// The column that is the left of the A tile,
		// and the row that is the top of the B tile
		const int I_TILE_START = i * TILE_SIZE;

		// Get pointers to the start of each tile, makes later indexing simpler
		const float* A_TILE = &A[A_TILE_ROW * numAColumns + I_TILE_START];
		const float* B_TILE = &B[I_TILE_START * numBColumns + B_TILE_COL];

		// Load A tile, checking for valid matrix row/col
		if (A_TILE_ROW + TILE_Y < numARows && I_TILE_START + TILE_X < numAColumns)
		{
			aTile[TILE_X][TILE_Y] = A_TILE[TILE_Y * numAColumns + TILE_X];
		}
		else 
		{
			aTile[TILE_X][TILE_Y] = 0;
		}

		// Load B tile, checking for valid matrix row/col
		if (I_TILE_START + TILE_Y < numBRows && B_TILE_COL + TILE_X < numBColumns)
		{
			bTile[TILE_X][TILE_Y] = B_TILE[TILE_Y * numBColumns + TILE_X];
		}
		else
		{
			bTile[TILE_X][TILE_Y] = 0;
		}
		
		__syncthreads();// Sync Reads

		if (VALID_C_IND)
		{
			// Compute the partial dot product across/down the tile
			for (int j = 0; j < TILE_SIZE; ++j)
			{
				partialDotProduct += aTile[j][TILE_Y] * bTile[TILE_X][j];
			}
		}
		__syncthreads();// Sync Writes
	}

	if (VALID_C_IND)
	{
		C[C_ROW * numCColumns + C_COL] = partialDotProduct;
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
