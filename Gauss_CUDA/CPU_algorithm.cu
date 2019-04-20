#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <cstring>

#include "Utils.h"

// For debug purposes
#define pAB Utils::printMatrix(matrixAB, degreeOfMatrixA, degreeOfMatrixA + 1)


// Operations are being conducted in-place
void solveLinearSystem(int degreeOfMatrixA, double **matrixAB) {
	int baseRowIdx;
	double columnDivider, subtractCoeff;
	//pAB;
	for (int colIdx = 0; colIdx < degreeOfMatrixA; ++colIdx) {
		// Choose base row
		baseRowIdx = -1;
		for (int rowIdx = colIdx; rowIdx < degreeOfMatrixA; ++rowIdx) {
			if (matrixAB[rowIdx][colIdx] != 0) {
				baseRowIdx = rowIdx;
				break;
			}
		}
		if (baseRowIdx == -1) {
			throw std::invalid_argument("Column can't contain zeros only");
		}
		// Exchange rows if columnDivider isn't matrixAB[colIdx][colIdx]
		else if (baseRowIdx != colIdx) {
			// Could be implemented as parallel operation (vector exchange)
			for (int exchColIdx = 0; exchColIdx < degreeOfMatrixA + 1; ++exchColIdx) {
				Utils::swap(matrixAB[baseRowIdx][exchColIdx], matrixAB[colIdx][exchColIdx]);
			}
			baseRowIdx = colIdx;
		}
		//pAB;
		// Chosen row's normalization
		columnDivider = matrixAB[baseRowIdx][colIdx];
		// Could be implemented as parallel operation (vector by constant division)
		for (int colNormIdx = colIdx; colNormIdx < degreeOfMatrixA + 1; ++colNormIdx) {
			matrixAB[baseRowIdx][colNormIdx] /= columnDivider;
			//pAB;
		}
		// Perform row subtraction
		// Could be implemented as parallel operation (multiple vectors subtraction)
		for (int rowSubIdx = 0; rowSubIdx < degreeOfMatrixA; ++rowSubIdx) {
			if (rowSubIdx == baseRowIdx || matrixAB[rowSubIdx][colIdx] == 0) {
				continue;
			}
			//pAB;
			subtractCoeff = matrixAB[rowSubIdx][colIdx];
			for (int colSubIdx = colIdx; colSubIdx < degreeOfMatrixA + 1; ++colSubIdx) {
				matrixAB[rowSubIdx][colSubIdx] -= subtractCoeff * matrixAB[baseRowIdx][colSubIdx];
				//pAB;
			}
		}
	}
}

int main(int argc, char *argv[]) {
	const int RANK = 4;
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
	double **matrixAB = new double*[RANK];
	for (int i = 0; i < RANK; ++i) {
		matrixAB[i] = new double[RANK + 1];
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

	double **matrixAB_copy = new double*[RANK];
	for (int i = 0; i < RANK; ++i) {
		matrixAB_copy[i] = new double[RANK + 1];
		std::memcpy(matrixAB_copy[i], matrixAB[i], (RANK + 1) * sizeof(double));
	}

	std::cout << "Matrix [A|B]:" << std::endl;
	Utils::printMatrix(matrixAB, RANK, RANK + 1);
	std::cout << std::endl;

	std::cout << "Copy of matrix [A|B]:" << std::endl;
	Utils::printMatrix(matrixAB_copy, RANK, RANK + 1);
	std::cout << std::endl;

	if (Utils::checkLinearSystem) {
		std::cout << "Created linear system is correct!" << std::endl;
	}
	else {
		std::cout << "Created linear system is incorrect!" << std::endl;
	}

	solveLinearSystem(RANK, matrixAB);
	std::cout << std::endl;
	std::cout << "Solved linear system:" << std::endl;
	Utils::printMatrix(matrixAB, RANK, RANK + 1);

	Utils::printSolutionVectorFromMatrix(RANK, matrixAB);
	std::cout << std::endl;
	if (Utils::checkLSSolution(RANK, matrixAB_copy, matrixAB)) {
		std::cout << "Solution vector is correct!" << std::endl;
	}
	else {
		std::cout << "Solution vector is incorrect! (small floating-point operations' errors are possible)" << std::endl;
		std::cout << "Error: " << Utils::getLSSolutionError(RANK, matrixAB_copy, matrixAB) << std::endl;
	}
	
	for (int i = 0; i < RANK; ++i) {
		delete[] matrixAB[i];
		delete matrixAB_copy[i];
	}
	delete[] matrixAB;
	delete[] matrixAB_copy;

	return 0;
}