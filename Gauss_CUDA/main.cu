#include <iostream>
#include <fstream>
#include <chrono>

#include "Utils.h"
#include "CPU_algorithm.cuh"
#include "GPU_algorithm.cuh"


int main(int argc, char *argv[]) {

	bool gpu_timeout(false);
	int max_gpu_time(15);
	int step(25);
	int size(0);
	double **matrix(nullptr), **matrix_copy(nullptr);
	auto start = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed;
	std::ofstream out;
	gpu_info info;

	std::cout << "GPU timeout[s]: ";
	std::cin >> max_gpu_time;
	std::cout << "Step size: ";
	std::cin >> step;

	out.open("log\\time.csv");
	if (!out.is_open()) {
		throw std::exception("time.csv not opened!");
	}
	out << "Stopieñ macierzy g³ownej uk³adu,Czas obliczeñ CPU,Czas obliczeñ GPU\n";
	for (size = step; !gpu_timeout; size += step) {
		std::cout << "Current size: " << size << std::endl;
		out << size << ',';
		Utils::DeleteMatrix(matrix, size - step);
		Utils::DeleteMatrix(matrix_copy, size - step);
		Utils::GenMatrix(&matrix, size);		
		matrix_copy = Utils::DuplicateMatrix(matrix, size, size + 1);

		start = std::chrono::high_resolution_clock::now();
		solveLinearSystem(size, matrix);
		finish = std::chrono::high_resolution_clock::now();
		elapsed = finish - start;
		out << elapsed.count();
		out << ',';

		info = solveLinearSystemParallel(size, matrix_copy);
		switch (info.result) {
		case SUCCESS:
			out << info.time;
			break;
		case CUDA_ERROR:
			out << "CUDA_ERROR";
			break;
		case ALGORITHM_ERROR:
			out << "ALGORITHM_ERROR";
			break;
		default:
			break;
		}
		if (info.time / 1000 >= max_gpu_time)
			gpu_timeout = true;
		out << '\n';
		if (info.result != Result::SUCCESS)
			break;
	}

	out.close();
	Utils::DeleteMatrix(matrix, size - step);
	Utils::DeleteMatrix(matrix_copy, size - step);
	return 0;
}
