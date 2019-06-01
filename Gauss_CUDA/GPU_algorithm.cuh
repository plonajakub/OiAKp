
#define defaultThreadsPerBlock 32

enum Result {
	SUCCESS, CUDA_ERROR, ALGORITHM_ERROR
};

struct gpu_info {
	Result result;
	float time;
};

gpu_info solveLinearSystemParallel(int degreeOfMatrixA, double **matrixAB);