#include <iostream>
#include <cstring>
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Utils.h"

#define getEl(matrix, type, pitch, row, col) (*((type*)((char*)matrix + pitch * row + sizeof(type) * col)))

enum Result {
	SUCCESS, CUDA_ERROR, ALGORITHM_ERROR
};

__global__ void swapMatrixRowsParallel(double *matrix, size_t pitch, int matrixDimX, int rowIdx1, int rowIdx2) {
	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (colIdx < matrixDimX) {
		double &el1 = getEl(matrix, double, pitch, rowIdx1, colIdx);
		double &el2 = getEl(matrix, double, pitch, rowIdx2, colIdx);
		double temp = el1;
		el1 = el2;
		el2 = temp;
	}
}

__global__ void divideMatrixRowByConstantParallel(double *matrix, size_t pitch, int matrixDimX, int rowIdx, double divisor) {
	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (colIdx < matrixDimX) {
		getEl(matrix, double, pitch, rowIdx, colIdx) /= divisor;
	}
}

__global__ void subtracMatrixRows(double *matrix, size_t pitch, int matrixDimX, int subtrahendRowIdx, int minuendRowIdx, double subtractCoeff) {
	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (colIdx < matrixDimX) {
		double &subtrahend = getEl(matrix, double, pitch, subtrahendRowIdx, colIdx);
		double &minuend = getEl(matrix, double, pitch, minuendRowIdx, colIdx);
		minuend -= subtractCoeff * subtrahend;
	}
}

__global__ void performMatrixVerticalRowSubtraction(double *matrix, size_t pitch, int matrixDimX, int matrixDimY, int baseRowIdx, int baseColIdx) {
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowIdx < matrixDimY && rowIdx != baseRowIdx && getEl(matrix, double, pitch, rowIdx, baseColIdx) != 0) {
		double subtractCoeff = getEl(matrix, double, pitch, rowIdx, baseColIdx);
		const int threadsPerBlock = 32;
		const int blocksPerGridDimX = ceil((matrixDimX) / (double)threadsPerBlock);
		cudaStream_t stream;
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		subtracMatrixRows << <blocksPerGridDimX, threadsPerBlock, 0, stream>> > (matrix, pitch, matrixDimX, baseRowIdx, rowIdx, subtractCoeff);
		cudaStreamDestroy(stream);
	}
}

__global__ void solveLinearSystemKernel(int degreeOfMatrixA, size_t pitch, double *matrixAB, Result &result_callback) {

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
			result_callback = Result::ALGORITHM_ERROR;
			return;
		}
		// Exchange rows if columnDivider isn't matrixAB[colIdx][colIdx]
		else if (baseRowIdx != colIdx) {
			// Exchange rows (vectors)
			swapMatrixRowsParallel << <blocksPerGridDimX, threadsPerBlock>> > (matrixAB, pitch, degreeOfMatrixA + 1, baseRowIdx, colIdx);
			baseRowIdx = colIdx;
		}

		cudaDeviceSynchronize();

		// Normalize choosen row
		columnDivider = getEl(matrixAB, double, pitch, baseRowIdx, colIdx);

		// Divide row (vector) by constant
		divideMatrixRowByConstantParallel << <blocksPerGridDimX, threadsPerBlock>> > (matrixAB, pitch, degreeOfMatrixA + 1, baseRowIdx, columnDivider);

		cudaDeviceSynchronize();

		// Perform multiple rows (vectors) subtraction
		performMatrixVerticalRowSubtraction << <blocksPerGridDimY, threadsPerBlock>> > (matrixAB, pitch, degreeOfMatrixA + 1, degreeOfMatrixA, baseRowIdx, colIdx);
	}
}

Result solveLinearSystemParallel(int degreeOfMatrixA, double **matrixAB) {
	// Error code holder
	cudaError_t cudaStatus;

	Result result = Result::SUCCESS;

	// Device memory allocation
	double *devMatrixAB;
	size_t pitch;
	cudaStatus = cudaMallocPitch(&devMatrixAB, &pitch,
		(degreeOfMatrixA + 1) * sizeof(double), degreeOfMatrixA);

	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMallocPitch failed!" << std::endl;
		result = Result::CUDA_ERROR;
		goto Cleanup;
	}

	// Copy memory from host to device
	for (int i = 0; i < degreeOfMatrixA; ++i) {
		cudaStatus = cudaMemcpy((void*)((char*)devMatrixAB + i * pitch), matrixAB[i],
			(degreeOfMatrixA + 1) * sizeof(double), cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMemcpy failed!" << std::endl;
			result = Result::CUDA_ERROR;
			goto Cleanup;
		}
	}

	// Kernel launch
	solveLinearSystemKernel << <1, 1 >> > (degreeOfMatrixA, pitch, devMatrixAB, result);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "solveLinearSystemKernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		result = Result::CUDA_ERROR;
		goto Cleanup;
	}

	// Wait for device to complete the work
	cudaStatus = cudaDeviceSynchronize();

	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaDeviceSynchronize returned error code " << cudaStatus <<
			" after launching solveLinearSystemKernel! " << std::endl;
		result = Result::CUDA_ERROR;
		goto Cleanup;
	}

	if (result == Result::ALGORITHM_ERROR) {
		std::cout << "Error during solving the linear system" << std::endl;
		goto Cleanup;
	}

	// Copy memory from device to host
	for (int i = 0; i < degreeOfMatrixA; ++i) {
		cudaStatus = cudaMemcpy(matrixAB[i], (void*)((char*)devMatrixAB + i * pitch),
			(degreeOfMatrixA + 1) * sizeof(double), cudaMemcpyDeviceToHost);

		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMemcpy failed!" << std::endl;
			result = Result::CUDA_ERROR;
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
		result = Result::CUDA_ERROR;
	}

	return result;
}


int main(int argc, char *argv[]) {

	// Matrix size is MATRIX_DEGREE x (MATRIX_DEGREE + 1)
	const int MATRIX_DEGREE = 4;

	/*double matrixAB[4][4] = {
		{2, 2, 2, 3},
		{3, 4, 5, 6},
		{2, 6, 5, 4},
		{1, 4, 1, 2}
	};
	double matrixB[4][1] = {
		{2},
		{4},
		{2},
		{2}
	};*/
	/*
		Solution:  -0.6
					0.1
				   -0.6
					1.4

	*/

	// Host allocation
	double **matrixAB = new double*[MATRIX_DEGREE];
	for (int i = 0; i < MATRIX_DEGREE; ++i) {
		matrixAB[i] = new double[MATRIX_DEGREE + 1];
	}

	// A
	matrixAB[0][0] = 2;
	matrixAB[0][1] = 2;
	matrixAB[0][2] = 2;
	matrixAB[0][3] = 3;

	matrixAB[1][0] = 3;
	matrixAB[1][1] = 4;
	matrixAB[1][2] = 5;
	matrixAB[1][3] = 6;

	matrixAB[2][0] = 2;
	matrixAB[2][1] = 6;
	matrixAB[2][2] = 5;
	matrixAB[2][3] = 4;

	matrixAB[3][0] = 1;
	matrixAB[3][1] = 4;
	matrixAB[3][2] = 1;
	matrixAB[3][3] = 2;

	// B
	matrixAB[0][4] = 2;
	matrixAB[1][4] = 4;
	matrixAB[2][4] = 2;
	matrixAB[3][4] = 2;

	// Create copy of matrixAB
	double **matrixAB_copy = new double*[MATRIX_DEGREE];
	for (int i = 0; i < MATRIX_DEGREE; ++i) {
		matrixAB_copy[i] = new double[MATRIX_DEGREE + 1];
		std::memcpy(matrixAB_copy[i], matrixAB[i], (MATRIX_DEGREE + 1) * sizeof(double));
	}

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
	Result result;
	result = solveLinearSystemParallel(MATRIX_DEGREE, matrixAB);

	switch (result) {
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

	// Host memory cleanup
	for (int i = 0; i < MATRIX_DEGREE; ++i) {
		delete[] matrixAB[i];
		delete matrixAB_copy[i];
	}
	delete[] matrixAB;
	delete[] matrixAB_copy;

	return 0;
}