#include <iostream>
#include <exception>
#include <iomanip>

// For debug purposes
#define pAB printMatrix(matrixAB, rankA, rankA + 1)

void printMatrix(double **matrix, int rowNum, int colNum) {
	std::cout << std::endl;
	for (int i = 0; i < rowNum; ++i) {
		std::cout << '[';
		for (int j = 0; j < colNum - 1; ++j) {
			std::cout << std::setw(4) << matrix[i][j] << ',';
		}
		std::cout << std::setw(4) << matrix[i][colNum - 1] << ']' << std::endl;
	}
}

template <class T>
void swap(T &a, T &b) {
	T temp = a;
	a = b;
	b = temp;
}

// Operations are being conducted in-place
void solveLinearSystem(double **matrixAB, int rankA) {
	int baseRowIdx;
	double columnDivider, subtractCoeff;
	pAB;
	for (int colIdx = 0; colIdx < rankA; ++colIdx) {
		// Choose base row
		baseRowIdx = -1;
		for (int rowIdx = colIdx; rowIdx < rankA; ++rowIdx) {
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
			for (int exchColIdx = 0; exchColIdx < rankA + 1; ++exchColIdx) {
				swap(matrixAB[baseRowIdx][exchColIdx], matrixAB[colIdx][exchColIdx]);
			}
			baseRowIdx = colIdx;
		}
		pAB;
		// Chosen row's normalization
		columnDivider = matrixAB[baseRowIdx][colIdx];
		// Could be implemented as parallel operation (vector by constant division)
		for (int colNormIdx = 0; colNormIdx < rankA + 1; ++colNormIdx) {
			matrixAB[baseRowIdx][colNormIdx] /= columnDivider;
			pAB;
		}
		// Perform row subtraction
		// Could be implemented as parallel operation (multiple vectors subtraction)
		for (int rowSubIdx = 0; rowSubIdx < rankA; ++rowSubIdx) {
			if (rowSubIdx == baseRowIdx || matrixAB[rowSubIdx][colIdx] == 0) {
				continue;
			}
			pAB;
			subtractCoeff = matrixAB[rowSubIdx][colIdx];
			for (int colSubIdx = 0; colSubIdx < rankA + 1; ++colSubIdx) {
				matrixAB[rowSubIdx][colSubIdx] -= subtractCoeff * matrixAB[baseRowIdx][colSubIdx];
				pAB;
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

	solveLinearSystem(matrixAB, RANK);
	std::cout << std::endl << "Solution vector:" << std::endl;
	std::cout << '[' << std::setw(4) << matrixAB[0][RANK] << std::endl;
	for (int i = 1; i < RANK - 1; ++i) {
		std::cout << std::setw(5) << matrixAB[i][RANK] << std::endl;
	}
	std::cout << std::setw(5) << matrixAB[RANK - 1][RANK] << " ]" << std::endl;

	for (int i = 0; i < RANK; ++i) {
		delete[] matrixAB[i];
	}
	delete[] matrixAB;
	return 0;
}