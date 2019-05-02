#include <iostream>
#include <cstring>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Utils.h"

#define getEl(matrix, type, pitch, row, col) (*((type*)((char*)matrix + pitch * row + sizeof(type) * col)))



__global__ void solveLinearSystemParallel(int degreeOfMatrixA, size_t pitch, double *matrixAB) {

	int baseRowIdx;
	double columnDivider, subtractCoeff;

	for (int colIdx = 0; colIdx < degreeOfMatrixA; ++colIdx) {

		// Choose base row
		baseRowIdx = -1;
		for (int rowIdx = colIdx; rowIdx < degreeOfMatrixA; ++rowIdx) {
			if (getEl(matrixAB, double, pitch, rowIdx, colIdx) != 0) {
				baseRowIdx = rowIdx;
				break;
			}
		}
		if (baseRowIdx == -1) {
			/*throw std::invalid_argument("Column can't contain zeros only");*/
			return;
		}
		// Exchange rows if columnDivider isn't matrixAB[colIdx][colIdx]
		else if (baseRowIdx != colIdx) {
			// Could be implemented as parallel operation (vector exchange)
			for (int exchColIdx = 0; exchColIdx < degreeOfMatrixA + 1; ++exchColIdx) {
				Utils::swap(getEl(matrixAB, double, pitch, baseRowIdx, exchColIdx),
					getEl(matrixAB, double, pitch, colIdx, exchColIdx));
			}
			baseRowIdx = colIdx;
		}

		// Normalize choosen row
		columnDivider = getEl(matrixAB, double, pitch, baseRowIdx, colIdx);
		// Could be implemented as parallel operation (vector by constant division)
		for (int colNormIdx = colIdx; colNormIdx < degreeOfMatrixA + 1; ++colNormIdx) {
			getEl(matrixAB, double, pitch, baseRowIdx, colNormIdx) /= columnDivider;
		}

		// Perform row subtraction
		// Could be implemented as parallel operation (multiple vectors subtraction)
		for (int rowSubIdx = 0; rowSubIdx < degreeOfMatrixA; ++rowSubIdx) {
			if (rowSubIdx == baseRowIdx || getEl(matrixAB, double, pitch, rowSubIdx, colIdx) == 0) {
				continue;
			}
			subtractCoeff = getEl(matrixAB, double, pitch, rowSubIdx, colIdx);
			for (int colSubIdx = colIdx; colSubIdx < degreeOfMatrixA + 1; ++colSubIdx) {
				getEl(matrixAB, double, pitch, rowSubIdx, colSubIdx) -= subtractCoeff * getEl(matrixAB, double, pitch, baseRowIdx, colSubIdx);
			}
		}
	}
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

	// Error code holder
	cudaError_t cudaStatus;

	// Device memory allocation
	double* devMatrixAB;
	size_t pitch;
	cudaStatus = cudaMallocPitch(&devMatrixAB, &pitch,
		(MATRIX_DEGREE + 1) * sizeof(double), MATRIX_DEGREE);

	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMallocPitch failed!" << std::endl;
		goto Error;
	}

	// Copy memory from host to device
	for (int i = 0; i < MATRIX_DEGREE; ++i) {
		cudaStatus = cudaMemcpy((void *)((char*)devMatrixAB + i * pitch), matrixAB[i],
			(MATRIX_DEGREE + 1) * sizeof(double), cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMemcpy failed!" << std::endl;
			goto Error;
		}
	}

	// Kernel lunch
	solveLinearSystemParallel<<<1, 1>>>(MATRIX_DEGREE, pitch, devMatrixAB);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "solveLinearSystemParallel kernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		goto Error;
	}

	// Wait for device to complete the work
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaDeviceSynchronize returned error code " << cudaStatus << 
			" after launching solveLinearSystemParallel! " << std::endl;
		goto Error;
	}

	// Copy memory from device to host
	for (int i = 0; i < MATRIX_DEGREE; ++i) {
		cudaStatus = cudaMemcpy(matrixAB[i], (void*)((char*)devMatrixAB + i * pitch),
			(MATRIX_DEGREE + 1) * sizeof(double), cudaMemcpyDeviceToHost);

		if (cudaStatus != cudaSuccess) {
			std::cout << "cudaMemcpy failed!" << std::endl;
			goto Error;
		}
	}

	// Device memory cleanup
	Error:
	cudaFree(devMatrixAB);

	// Device reset for profiling tools
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaDeviceReset failed!" << std::endl;
		return 1;
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