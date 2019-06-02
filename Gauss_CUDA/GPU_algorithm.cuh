#pragma once

enum Result {
	SUCCESS, CUDA_ERROR, ALGORITHM_ERROR
};

struct gpu_info {
	Result result;
	float time;
};

gpu_info solveLinearSystemParallel(int threadsPerBlock, int degreeOfMatrixA, double **matrixAB);