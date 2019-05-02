#pragma once

#include <cmath>
#include <stdexcept>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


class Utils {
public:
	Utils();
	~Utils();

	/* Checks if provided linear system has
	 * only one solution vector (calculates rank of matrix [A|B])
	 *
	 * matrixAB is concatenated matrix A and B ([A|B])
	 *
	 * Returns true if LS has one and only one solution vector, otherwise returns false
	 */
	static bool checkLinearSystem(int degreeOfMatrixA, double **matrixAB) {
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
				// It is possible for LS to be correct here
				// but matrix A doesn't have provided degree
				return false;
			}
			// Exchange rows if columnDivider isn't matrixAB[colIdx][colIdx]
			else if (baseRowIdx != colIdx) {
				// Could be implemented as parallel operation (vector exchange)
				for (int exchColIdx = 0; exchColIdx < degreeOfMatrixA + 1; ++exchColIdx) {
					swap(matrixAB[baseRowIdx][exchColIdx], matrixAB[colIdx][exchColIdx]);
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
			for (int rowSubIdx = baseRowIdx + 1; rowSubIdx < degreeOfMatrixA; ++rowSubIdx) {
				if (matrixAB[rowSubIdx][colIdx] == 0) {
					continue;
				}
				subtractCoeff = matrixAB[rowSubIdx][colIdx];
				for (int colSubIdx = colIdx; colSubIdx < degreeOfMatrixA + 1; ++colSubIdx) {
					matrixAB[rowSubIdx][colSubIdx] -= subtractCoeff * matrixAB[baseRowIdx][colSubIdx];
				}
			}
		}

		// Check for zeros only rows
		bool nonZeroValuePresent;
		for (int rowIdx = 0; rowIdx < degreeOfMatrixA; ++rowIdx) {
			nonZeroValuePresent = false;
			for (int colIdx = 0; colIdx < degreeOfMatrixA; ++colIdx) {
				if (matrixAB[rowIdx][colIdx] != 0) {
					nonZeroValuePresent = true;
					break;
				}
			}
			if (nonZeroValuePresent) {
				continue;
			}
			else {
				return false;
			}
		}
		return true;
	}

	/*
	 * Calulates error for provided linear system's solution
	 * 
	 * solution has the same size as matrixAB and is in form of [I|X]
	 * 
	 * Returns absolute error (sum of each equation's absolute error)
	 */
	template <class T>
	static double getLSSolutionError(int degreeOfMatrixA, T **matrixAB, T **solution) {
		double equationValue;
		double error = 0;
		for (int rowIdx = 0; rowIdx < degreeOfMatrixA; ++rowIdx) {
			equationValue = 0;
			for (int colIdx = 0; colIdx < degreeOfMatrixA; ++colIdx) {
				equationValue += matrixAB[rowIdx][colIdx] * solution[colIdx][degreeOfMatrixA];
			}
			error += fabs(matrixAB[rowIdx][degreeOfMatrixA] - equationValue);
		}
		return error;
	}

	/*
	 * Checks if provided linear system's solution is correct
	 *
	 * solution has the same size as matrixAB and is in form of [I|X]
	 *
	 * Returns true if solution is correct, otherwise returns false
	 */
	template <class T>
	static bool checkLSSolution(int degreeOfMatrixA, T **matrixAB, T **solution) {
		double equationValue;
		double error;
		for (int rowIdx = 0; rowIdx < degreeOfMatrixA; ++rowIdx) {
			equationValue = 0;
			for (int colIdx = 0; colIdx < degreeOfMatrixA; ++colIdx) {
				equationValue += matrixAB[rowIdx][colIdx] * solution[colIdx][degreeOfMatrixA];
			}
			error = matrixAB[rowIdx][degreeOfMatrixA] - equationValue;
			if (error != 0) {
				return false;
			}
		}
		return true;
	}

	
	template <class T>
	__host__ __device__ static void swap(T &a, T &b) {
		T temp = a;
		a = b;
		b = temp;
	}

	template <class T>
	static void printMatrix(T **matrix, int rowNum, int colNum) {
		for (int i = 0; i < rowNum; ++i) {
			std::cout << '[';
			for (int j = 0; j < colNum - 1; ++j) {
				std::cout << std::setw(4) << matrix[i][j] << ',';
			}
			std::cout << std::setw(4) << matrix[i][colNum - 1] << ']' << std::endl;
		}
	}

	template <class T>
	static void printSolutionVectorFromMatrix(int degreeOfMatrixA, T **matrixAB) {
		std::cout << std::endl << "Solution vector:" << std::endl;
		std::cout << '[' << std::setw(4) << matrixAB[0][degreeOfMatrixA] << std::endl;
		for (int i = 1; i < degreeOfMatrixA - 1; ++i) {
			std::cout << std::setw(5) << matrixAB[i][degreeOfMatrixA] << std::endl;
		}
		std::cout << std::setw(5) << matrixAB[degreeOfMatrixA - 1][degreeOfMatrixA] << " ]" << std::endl;
	}
};

