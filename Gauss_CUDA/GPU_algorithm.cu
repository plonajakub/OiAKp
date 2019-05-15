#include <iostream>
#include <cstring>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include "Utils.h"
#include "GPU_algorithm.cuh"

#define getEl(matrix, type, pitch, row, col) (*((type*)((char*)matrix + pitch * row + sizeof(type) * col)))
#define defaultThreadsPerBlock (32)

__global__ void swapMatrixRowsParallel(double *matrix, size_t pitch, int matrixDimX, int rowIdx1, int rowIdx2) {
	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ double row[];
	if (colIdx < matrixDimX) {
		row[threadIdx.x] = getEl(matrix, double, pitch, rowIdx1, colIdx);
		getEl(matrix, double, pitch, rowIdx1, colIdx) = getEl(matrix, double, pitch, rowIdx2, colIdx);
		getEl(matrix, double, pitch, rowIdx2, colIdx) = row[threadIdx.x];
	}
}

__global__ void divideMatrixRowByConstantParallel(double *matrix, size_t pitch, int matrixDimX, int rowIdx, double divisor) {
	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ double row[];
	if (colIdx < matrixDimX) {
		row[threadIdx.x] = getEl(matrix, double, pitch, rowIdx, colIdx);
		row[threadIdx.x] /= divisor;
		getEl(matrix, double, pitch, rowIdx, colIdx) = row[threadIdx.x];
	}
}

__global__ void subtracMatrixRows(double *matrix, size_t pitch, int matrixDimX, int subtrahendRowIdx, int minuendRowIdx, double subtractCoeff) {
	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ double s[];
	if (colIdx < matrixDimX) {
		double *minRow = s, *subRow = s + blockDim.x;
		minRow[threadIdx.x] = getEl(matrix, double, pitch, minuendRowIdx, colIdx);
		subRow[threadIdx.x] = getEl(matrix, double, pitch, subtrahendRowIdx, colIdx); // Conflicts in memory can occur here
		minRow[threadIdx.x] -= subtractCoeff * subRow[threadIdx.x];
		getEl(matrix, double, pitch, minuendRowIdx, colIdx) = minRow[threadIdx.x];
	}
}

__global__ void performMatrixVerticalRowSubtraction(double *matrix, size_t pitch, int matrixDimX, int matrixDimY, int baseRowIdx, int baseColIdx) {
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ double subtracCoeffs[];
	if (rowIdx < matrixDimY && rowIdx != baseRowIdx) {
		subtracCoeffs[threadIdx.x] = getEl(matrix, double, pitch, rowIdx, baseColIdx);
		if (subtracCoeffs[threadIdx.x] != 0) {
			const int blocksPerGridDimX = ceil((matrixDimX) / (double)defaultThreadsPerBlock);
			cudaStream_t stream;
			cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
			subtracMatrixRows<<<blocksPerGridDimX, defaultThreadsPerBlock, 2 * defaultThreadsPerBlock * sizeof(double), stream>>>
				(matrix, pitch, matrixDimX, baseRowIdx, rowIdx, subtracCoeffs[threadIdx.x]);
			cudaStreamDestroy(stream);
		}
	}
}

__global__ void solveLinearSystemKernel(int degreeOfMatrixA, size_t pitch, double *matrixAB, Result &outputData) {

	const int threadsPerBlock = 32;
	const int blocksPerGridDimX = ceil((degreeOfMatrixA + 1) / (double)threadsPerBlock);
	const int blocksPerGridDimY = ceil(degreeOfMatrixA / (double)threadsPerBlock);

	int baseRowIdx;
	double columnDivider;

	for (int colIdx = 0; colIdx < degreeOfMatrixA; ++colIdx) {

		// Choose base row
		baseRowIdx = -1;
		cudaDeviceSynchronize();
		for (int rowIdx = colIdx; rowIdx < degreeOfMatrixA; ++rowIdx) {
			if (getEl(matrixAB, double, pitch, rowIdx, colIdx) != 0) {
				baseRowIdx = rowIdx;
				break;
			}
		}
		if (baseRowIdx == -1) {
			outputData = Result::ALGORITHM_ERROR;
			return;
		}
		// Exchange rows if columnDivider isn't matrixAB[colIdx][colIdx]
		else if (baseRowIdx != colIdx) {
			// Exchange rows (vectors)
			swapMatrixRowsParallel <<<blocksPerGridDimX, threadsPerBlock, 2 * threadsPerBlock * sizeof(double)>>>
				(matrixAB, pitch, degreeOfMatrixA + 1, baseRowIdx, colIdx);
			baseRowIdx = colIdx;
		}

		cudaDeviceSynchronize();

		// Normalize choosen row
		columnDivider = getEl(matrixAB, double, pitch, baseRowIdx, colIdx);

		// Divide row (vector) by constant
		divideMatrixRowByConstantParallel<<<blocksPerGridDimX, threadsPerBlock, threadsPerBlock * sizeof(double)>>>
			(matrixAB, pitch, degreeOfMatrixA + 1, baseRowIdx, columnDivider);

		cudaDeviceSynchronize();

		// Perform multiple rows (vectors) subtraction
		performMatrixVerticalRowSubtraction<<<blocksPerGridDimY, threadsPerBlock, threadsPerBlock * sizeof(double)>>>
			(matrixAB, pitch, degreeOfMatrixA + 1, degreeOfMatrixA, baseRowIdx, colIdx);
	}
}

gpu_info solveLinearSystemParallel(int degreeOfMatrixA, double **matrixAB) {
	// Error code holder
	cudaError_t cudaStatus;

	// Configuration of shared memory banks' size
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	// Returned data
	gpu_info info = {Result::SUCCESS, 0.0};
	
	// Algorithm's work time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Device memory allocation (pitched memory)
	double *devMatrixAB;
	size_t pitch;
	cudaStatus = cudaMallocPitch(&devMatrixAB, &pitch,
		(degreeOfMatrixA + 1) * sizeof(double), degreeOfMatrixA);

	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMallocPitch failed!" << std::endl;
		std::cout << "Error description: " << cudaGetErrorString(cudaStatus) << std::endl;
		info.result = Result::CUDA_ERROR;
		goto Cleanup;
	}

	// Copy memory from host to device
	for (int i = 0; i < degreeOfMatrixA; ++i) {
		cudaStatus = cudaMemcpy((void*)((char*)devMatrixAB + i * pitch), matrixAB[i],
			(degreeOfMatrixA + 1) * sizeof(double), cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMemcpy failed!" << std::endl;
			std::cout << "Error description: " << cudaGetErrorString(cudaStatus) << std::endl;
			info.result = Result::CUDA_ERROR;
			goto Cleanup;
		}
	}

	//start time measurement
	cudaEventRecord(start);

	// Kernel launch
	solveLinearSystemKernel<<<1, 1>>>(degreeOfMatrixA, pitch, devMatrixAB, info.result); 

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "solveLinearSystemKernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		std::cout << "Error description: " << cudaGetErrorString(cudaStatus) << std::endl;
		info.result = Result::CUDA_ERROR;
		goto Cleanup;
	}

	// Wait for device to complete the work
	cudaStatus = cudaDeviceSynchronize();

	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaDeviceSynchronize returned error code " << cudaStatus <<
			" after launching solveLinearSystemKernel! " << std::endl;
		info.result = Result::CUDA_ERROR;
		goto Cleanup;
	}

	//stop time measurement
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&info.time, start, stop);

	if (info.result == Result::ALGORITHM_ERROR) {
		std::cout << "Provided linear system hasn't proper format!" << std::endl;
		goto Cleanup;
	}

	// Copy memory from device to host
	for (int i = 0; i < degreeOfMatrixA; ++i) {
		cudaStatus = cudaMemcpy(matrixAB[i], (void*)((char*)devMatrixAB + i * pitch),
			(degreeOfMatrixA + 1) * sizeof(double), cudaMemcpyDeviceToHost);

		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMemcpy failed!" << std::endl;
			std::cout << "Error description: " << cudaGetErrorString(cudaStatus) << std::endl;
			info.result = Result::CUDA_ERROR;
			goto Cleanup;
		}
	}

Cleanup:
	// Device memory cleanup
	cudaFree(devMatrixAB);

	// Device reset for profiling tools
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaDeviceReset failed!" << std::endl;
		std::cout << "Error description: " << cudaGetErrorString(cudaStatus) << std::endl;
		info.result = Result::CUDA_ERROR;
	}
	return info;
}


int main(int argc, char *argv[]) {
	const int MATRIX_DEGREE = 5;

	// Host allocation
	double **matrixAB = nullptr;
	Utils::GenMatrix(&matrixAB, MATRIX_DEGREE, 0);

	// Create copy of matrixAB
	double **matrixAB_copy = nullptr;
	matrixAB_copy = Utils::DuplicateMatrix(matrixAB, MATRIX_DEGREE, MATRIX_DEGREE + 1);

	// Print generated matrix
	std::cout << "Matrix [A|B]:" << std::endl;
	Utils::printMatrix(matrixAB, MATRIX_DEGREE, MATRIX_DEGREE + 1);
	std::cout << std::endl;

	std::cout << "Copy of matrix [A|B]:" << std::endl;
	Utils::printMatrix(matrixAB_copy, MATRIX_DEGREE, MATRIX_DEGREE + 1);
	std::cout << std::endl;

	if (Utils::checkLinearSystem(MATRIX_DEGREE, matrixAB)) {
		std::cout << "Created linear system is correct!" << std::endl;
	}
	else {
		std::cout << "Created linear system is incorrect!" << std::endl;
	}

	// Solve linear system on GPU
	gpu_info info;
	info = solveLinearSystemParallel(MATRIX_DEGREE, matrixAB);

	switch (info.result) {
	case SUCCESS:
		std::cout << "solverLinearSystemParallel() returned without errors!" << std::endl;
		break;
	case CUDA_ERROR:
		std::cout << "solverLinearSystemParallel() returned cuda error!" << std::endl;
		break;
	case ALGORITHM_ERROR:
		std::cout << "solverLinearSystemParallel() returned algorithm error!" << std::endl;
		break;
	default:
		break;
	}

	// Present results
	std::cout << std::endl;
	std::cout << "Solved linear system:" << std::endl;
	Utils::printMatrix(matrixAB, MATRIX_DEGREE, MATRIX_DEGREE + 1);

	Utils::printSolutionVectorFromMatrix(MATRIX_DEGREE, matrixAB);
	std::cout << std::endl;

	if (Utils::checkLSSolution(MATRIX_DEGREE, matrixAB_copy, matrixAB)) {
		std::cout << "Solution vector is correct!" << std::endl;
	}
	else {
		std::cout << "Solution vector is incorrect! (small floating-point operations' errors are possible)" << std::endl;
		std::cout << "Error: " << Utils::getLSSolutionError(MATRIX_DEGREE, matrixAB_copy, matrixAB) << std::endl;
	}
	
	std::cout << "Calculations time: " << info.time << " ms" << std::endl;

	// Host memory cleanup
	Utils::DeleteMatrix(matrixAB, MATRIX_DEGREE);
	Utils::DeleteMatrix(matrixAB_copy, MATRIX_DEGREE);

	return 0;
}
