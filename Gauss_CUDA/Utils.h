#pragma once

#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define min(a, b) ((b >= a) ? a : b)


class Utils {
public:
	Utils();
	~Utils();

	class UniRand
	{
	public:
		static UniRand &Get() {
			static UniRand obj;
			return obj;
		};
		int RandInt(int from, int to) {
			return std::uniform_int_distribution<int>{from, to}(mt);
		};
		double RandDouble(double from, double to) {
			return std::uniform_real_distribution<double>{from, to}(mt);
		};
		std::mt19937 &Generator() {
			return mt;
		};
	private:
		UniRand()
			: mt(std::random_device()())
		{};
		UniRand(UniRand const&);
		void operator=(UniRand const&);

		std::mt19937 mt;
	};

	static void DeleteMatrix(double **matrix, int size) {
		if (matrix == nullptr || size < 1)
			return;

		for (int i(size - 1); i >= 0; i--)
			delete[] matrix[i];
		delete matrix;
	}

	static double **DuplicateMatrix(double **matrix, int row_size, int col_size) {
		double **cp = new double*[row_size];

		for (int i(0); i < row_size; i++)
		{
			cp[i] = new double[col_size];
			for (int j(0); j < col_size; j++)
				cp[i][j] = matrix[i][j];
		}

		return cp;
	}

	static bool Determinant(double **matrix, int size) {
		bool solvable(true);
		double **mult = DuplicateMatrix(matrix, size, size);

		int max;
		double m;
		for (int i(0); i < size - 1; i++)
		{
			//place highest val from M = {m[x][i]: x = <i, size-2>} into m[i][i]
			max = i;
			for (int j(i); j < size; j++)
				if (abs(mult[j][i]) > abs(mult[max][i]))
					max = j;
			//swap
			swap(mult[max], mult[i]);

			//for every row reduce every row bellow
			for (int j(i + 1); j < size; j++)
			{
				//max element(diagonal) == 0 => det == 0
				if (mult[i][i] == 0)
				{
					solvable = false;
					break;
				}
				//already reduced
				if (mult[j][i] == 0)
					continue;

				m = mult[j][i] / mult[i][i];
				for (int k(min(j + 1, size - 1)); k >= i; k--)
					mult[j][k] -= m * mult[i][k];
			}
			if (!solvable)
				break;
		}
		if (abs(mult[size - 1][size - 1]) < 0.001)
			solvable = false;

		/*
		printMatrix(st_matrix{ mult , size });
		double det = mult[0][0];
		for (int i(1); i < size; i++)
			det *= mult[i][i];
		if (det < 0.1 && det > -0.1)
			det = 0;
		std::cout << "det: " << det << std::endl;
		/**/

		DeleteMatrix(mult, size);

		return solvable;
	}

	//generate random matrix with one solution
	static void GenMatrix(double ***matrix, int size_new) {
		(*matrix) = new double*[size_new];
		for (int i(0); i < size_new; i++)
			(*matrix)[i] = new double[size_new + 1];

		do {
			for (int i(0); i < size_new; i++)
			{
				for (int j(0); j <= size_new; j++)
				{
					(*matrix)[i][j] = UniRand::Get().RandInt(1, 100000);
				}
			}
		} while (!Determinant(*matrix, size_new));
	}

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
		const int fieldWidth = 10;
		for (int i = 0; i < rowNum; ++i) {
			std::cout << '[';
			for (int j = 0; j < colNum - 1; ++j) {
				std::cout << std::setw(fieldWidth) << matrix[i][j] << ',';
			}
			std::cout << std::setw(fieldWidth) << matrix[i][colNum - 1] << ']' << std::endl;
		}
	}

	template <class T>
	static void printSolutionVectorFromMatrix(int degreeOfMatrixA, T **matrixAB) {
		const int fieldWidth = 11;
		std::cout << std::endl << "Solution vector:" << std::endl;
		std::cout << '[' << std::setw(fieldWidth - 1) << matrixAB[0][degreeOfMatrixA] << std::endl;
		for (int i = 1; i < degreeOfMatrixA - 1; ++i) {
			std::cout << std::setw(fieldWidth) << matrixAB[i][degreeOfMatrixA] << std::endl;
		}
		std::cout << std::setw(fieldWidth) << matrixAB[degreeOfMatrixA - 1][degreeOfMatrixA] << " ]" << std::endl;
	}
};

