
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include "Utils.h"
#include "CPU_algorithm.cuh"
#include "GPU_algorithm.cuh"

//*
int main(int argc, char *argv[]) {
	const int k_step(100);
	const int k_max_cpu_time(120);
	const int k_max_gpu_time(120);
	bool cpu_timeout(false);
	bool gpu_timeout(false);
	int size(0);
	double **matrix(nullptr), **matrix_copy(nullptr);
	auto start = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed;
	std::ofstream out;
	gpu_info info;

	if(size != 0)
		Utils::GenMatrix(&matrix, size, 0);

	out.open("log\\time.csv");
	out << "size, cpu, gpu\n";
	while ((!cpu_timeout || !gpu_timeout) && (size += k_step))
	{
		std::cout << "size: " << size << std::endl;
		out << size << ',';
		Utils::GenMatrix(&matrix, size, size - k_step);
		Utils::DeleteMatrix(matrix_copy, size - k_step);
		matrix_copy = Utils::DuplicateMatrix(matrix, size, size + 1);

		if (!cpu_timeout)
		{
			start = std::chrono::high_resolution_clock::now();
			solveLinearSystem(size, matrix);
			finish = std::chrono::high_resolution_clock::now();
			elapsed = finish - start;
			out << elapsed.count();
			if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= k_max_cpu_time)
				cpu_timeout = true;
		}
		out << ',';

		if (!gpu_timeout)
		{
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
	}

	out.close();
	Utils::DeleteMatrix(matrix, size - k_step);
	Utils::DeleteMatrix(matrix_copy, size - k_step);
	return 0;
}
/**/