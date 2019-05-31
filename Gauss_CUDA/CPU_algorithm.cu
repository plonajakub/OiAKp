#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <cstring>
#include <chrono>

#include "Utils.h"
#include "CPU_algorithm.cuh"


// Operations are being conducted in-place
void solveLinearSystem(int degreeOfMatrixA, double **matrixAB) {
	int baseRowIdx;
	double columnDivider, subtractCoeff;

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

		// Chosen row's normalization
		columnDivider = matrixAB[baseRowIdx][colIdx];
		// Could be implemented as parallel operation (vector by constant division)
		for (int colNormIdx = colIdx; colNormIdx < degreeOfMatrixA + 1; ++colNormIdx) {
			matrixAB[baseRowIdx][colNormIdx] /= columnDivider;
		}

		// Perform row subtraction
		// Could be implemented as parallel operation (multiple vectors subtraction)
		for (int rowSubIdx = 0; rowSubIdx < degreeOfMatrixA; ++rowSubIdx) {
			if (rowSubIdx == baseRowIdx || matrixAB[rowSubIdx][colIdx] == 0) {
				continue;
			}
			subtractCoeff = matrixAB[rowSubIdx][colIdx];
			for (int colSubIdx = colIdx; colSubIdx < degreeOfMatrixA + 1; ++colSubIdx) {
				matrixAB[rowSubIdx][colSubIdx] -= subtractCoeff * matrixAB[baseRowIdx][colSubIdx];
			}
		}
	}
}

//int main(int argc, char *argv[]) {
//	const int MATRIX_DEGREE = 5;
//
//	// Create a linear system
//	double **matrixAB = nullptr;
//	Utils::GenMatrix(&matrixAB, MATRIX_DEGREE);
//
//	// Store copy of the linear system
//	double **matrixAB_copy = Utils::DuplicateMatrix(matrixAB, MATRIX_DEGREE, MATRIX_DEGREE + 1);
//
//	// Show created linear system
//	std::cout << "Matrix [A|B]:" << std::endl;
//	Utils::printMatrix(matrixAB, MATRIX_DEGREE, MATRIX_DEGREE + 1);
//	std::cout << std::endl;
//
//	std::cout << "Copy of matrix [A|B]:" << std::endl;
//	Utils::printMatrix(matrixAB_copy, MATRIX_DEGREE, MATRIX_DEGREE + 1);
//	std::cout << std::endl;
//
//	if (Utils::checkLinearSystem(MATRIX_DEGREE, matrixAB)) {
//		std::cout << "Created linear system is correct!" << std::endl;
//	}
//	else {
//		std::cout << "Created linear system is incorrect!" << std::endl;
//	}
//
//	auto start = std::chrono::high_resolution_clock::now();
//	auto end = std::chrono::high_resolution_clock::now();
//	std::chrono::duration<double, std::milli> elapsed;
//
//	start = std::chrono::high_resolution_clock::now();
//	
//	// Solve the linear system
//	solveLinearSystem(MATRIX_DEGREE, matrixAB);
//	
//	end = std::chrono::high_resolution_clock::now();
//	elapsed = end - start;
//
//	// Show results
//	std::cout << std::endl;
//	std::cout << "Solved linear system:" << std::endl;
//	Utils::printMatrix(matrixAB, MATRIX_DEGREE, MATRIX_DEGREE + 1);
//	Utils::printSolutionVectorFromMatrix(MATRIX_DEGREE, matrixAB);
//	std::cout << std::endl;
//
//	if (Utils::checkLSSolution(MATRIX_DEGREE, matrixAB_copy, matrixAB)) {
//		std::cout << "Solution vector is correct!" << std::endl;
//	}
//	else {
//		std::cout << "Solution vector is incorrect! (small floating-point operations' errors are possible)" << std::endl;
//		std::cout << "Error: " << Utils::getLSSolutionError(MATRIX_DEGREE, matrixAB_copy, matrixAB) << std::endl;
//	}
//
//	std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;
//	
//	// Cleanup
//	Utils::DeleteMatrix(matrixAB, MATRIX_DEGREE);
//	Utils::DeleteMatrix(matrixAB_copy, MATRIX_DEGREE);
//
//	return 0;
//}