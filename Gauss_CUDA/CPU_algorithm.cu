#include <iostream>
#include <exception>

// For debug purposes
#define pA printMatrix(matrixA, rank, rank);
#define pB printMatrix(matrixB, rank, 1);

void printMatrix(double **matrix, int rowNum, int colNum) {
	std::cout << std::endl;
	for (int i = 0; i < rowNum; ++i) {
		std::cout << '[';
		for (int j = 0; j < colNum - 1; ++j) {
			std::cout << matrix[i][j] << ',';
		}
		std::cout << matrix[i][colNum - 1] << ']' << std::endl;
	}
}

template <class T>
void swap(T &a, T &b) {
	T temp = a;
	a = b;
	b = temp;
}

void solveLinearSystem(double **matrixA, double **matrixB, int rank) {
	int baseRowIdx;
	double columnDivider, subtractCoeff;
	pA pB	
	for (int colIdx = 0; colIdx < rank; ++colIdx) {
		// Choose base row
		baseRowIdx = -1;
		for (int rowIdx = colIdx; rowIdx < rank; ++rowIdx) {
			if (matrixA[rowIdx][colIdx] != 0) {
				baseRowIdx = rowIdx;
				break;
			}
		}
		if (baseRowIdx == -1) {
			throw std::invalid_argument("Column can't contain only zeros");
		}
		// Exchange rows if columnDivider isn't matriX[colIdx][colIdx]
		else if (baseRowIdx != colIdx) {
			for (int exchColIdx = 0; exchColIdx < rank; ++exchColIdx) {
				swap(matrixA[baseRowIdx][colIdx], matrixA[colIdx][colIdx]);
			}
			swap(matrixB[baseRowIdx][0], matrixA[colIdx][0]);
			baseRowIdx = colIdx;
		}
		// Chosen row's normalization
		pA
		columnDivider = matrixA[baseRowIdx][colIdx];
		for (int colNormIdx = 0; colNormIdx < rank; ++colNormIdx) {
			matrixA[baseRowIdx][colNormIdx] /= columnDivider;
			pA
		}
		matrixB[baseRowIdx][0] /= columnDivider;
		pB
		// Perform row subtraction
		for (int rowSubIdx = 0; rowSubIdx < rank; ++rowSubIdx) {
			if (rowSubIdx == baseRowIdx || matrixA[rowSubIdx][colIdx] == 0) {
				continue;
			}
			pA
			subtractCoeff = matrixA[rowSubIdx][colIdx];
			for (int colSubIdx = 0; colSubIdx < rank; ++colSubIdx) {
				matrixA[rowSubIdx][colSubIdx] -= subtractCoeff * matrixA[baseRowIdx][colSubIdx];
				pA
			}
			matrixB[rowSubIdx][0] -= subtractCoeff * matrixB[baseRowIdx][0];
			pB
		}
	}
}

int main(int argc, char *argv[]) {
	const int RANK = 4;
	/*double matrixA[3][3] = {
		{2, 2, 2, 3},
		{3, 4, 5, 6},
		{2, 6, 5, 4},
		{1, 4, 1, 2}
	};
	double matrixB[3][1] = {
		{2},
		{4},
		{2},
		{2}
	};*/
	/*
		Solution:	-0.6
					0.1
					0.6
					1.4
	
	*/
	double **matrixA = new double*[RANK];
	double **matrixB = new double*[RANK];
	for (int i = 0; i < RANK; ++i) {
		matrixA[i] = new double[RANK];
		matrixB[i] = new double;
	}

	matrixA[0][0] = 2;
	matrixA[0][1] = 2;
	matrixA[0][2] = 2;
	matrixA[0][3] = 3;

	matrixA[1][0] = 3;
	matrixA[1][1] = 4;
	matrixA[1][2] = 5;
	matrixA[1][3] = 6;

	matrixA[2][0] = 2;
	matrixA[2][1] = 6;
	matrixA[2][2] = 5;
	matrixA[2][3] = 4;

	matrixA[3][0] = 1;
	matrixA[3][1] = 4;
	matrixA[3][2] = 1;
	matrixA[3][3] = 2;

	matrixB[0][0] = 2;
	matrixB[1][0] = 4;
	matrixB[2][0] = 2;
	matrixB[3][0] = 2;

	solveLinearSystem(matrixA, matrixB, RANK);
	std::cout << std::endl << "Solution vector: ";
	printMatrix(matrixB, RANK, 1);
	return 0;
}