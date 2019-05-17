#include <iostream>
#include <fstream>
#include <chrono>

#include "Utils.h"
#include "CPU_algorithm.cuh"
#include "GPU_algorithm.cuh"


int main(int argc, char *argv[]) {
	const int k_step(25);
	const int k_max_cpu_time(2);
	const int k_max_gpu_time(1);

	bool cpu_timeout(false);
	bool gpu_timeout(false);
	int size(0);
	double **matrix(nullptr), **matrix_copy(nullptr);
	auto start = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed;
	std::ofstream out;
	gpu_info info;

	out.open("log\\time.csv");
	if (!out.is_open()) {
		throw std::exception("time.csv not opened!");
	}
	out << "Stopieñ macierzy g³ownej uk³adu,Czas obliczeñ CPU,Czas obliczeñ GPU\n";
	for (size = k_step; !(cpu_timeout && gpu_timeout); size += k_step) {
		std::cout << "Current step size: " << size << std::endl;
		out << size << ',';
		Utils::DeleteMatrix(matrix, size - k_step);
		Utils::DeleteMatrix(matrix_copy, size - k_step);
		Utils::GenMatrix(&matrix, size);		
		matrix_copy = Utils::DuplicateMatrix(matrix, size, size + 1);

		if (!cpu_timeout) {
			start = std::chrono::high_resolution_clock::now();
			solveLinearSystem(size, matrix);
			finish = std::chrono::high_resolution_clock::now();
			elapsed = finish - start;
			out << elapsed.count();
			if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= k_max_cpu_time)
				cpu_timeout = true;
		}
		out << ',';

		if (!gpu_timeout) {
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
			if (info.time/1000 >= k_max_gpu_time)
				gpu_timeout = true;
		}
		out << '\n';
		if (info.result != Result::SUCCESS)
			break;
	}

	out.close();
	Utils::DeleteMatrix(matrix, size - k_step);
	Utils::DeleteMatrix(matrix_copy, size - k_step);
	return 0;
}
