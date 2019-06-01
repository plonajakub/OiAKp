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
	int size(5);

	double **matrix(nullptr), **matrix_copy(nullptr);
	auto start = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed;
	std::ofstream out;
	gpu_info info;

	std::cout << "GPU timeout[s]: ";
	std::cin >> max_gpu_time;
	std::cout << "Starting size: ";
	std::cin >> size;
	std::cout << "Step size: ";
	std::cin >> step;

	std::cout << std::endl << "Running..." << std::endl;

	out.open("log\\gauss_time.csv");
	if (!out.is_open()) {
		throw std::exception("gauss_time.csv not opened!");
	}
	out << "Rozmiar pocz\u0105tkowy = " << size << ",Krok = " << step << ",Limit czasu GPU = "
		<< max_gpu_time << "s,Liczba w\u0105tk\u00F3w w bloku = " << defaultThreadsPerBlock << std::endl;
	out << "Stopie\u0144 macierzy g\u0142ownej uk\u0142adu,Czas oblicze\u0144 CPU [s],Czas oblicze\u0144 GPU [s]" << std::endl;
	for (; !gpu_timeout; size += step) {
		std::cout << "Current matrix degree: " << size << std::endl;
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
			out << info.time / 1000;
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
		out << std::endl;
		if (info.result != Result::SUCCESS)
			break;
	}

	out.close();
	Utils::DeleteMatrix(matrix, size - step);
	Utils::DeleteMatrix(matrix_copy, size - step);
	return 0;
}
