#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

// Заполнение массива рандомными числами
void randomizeArray(std::vector<int>& array, std::mt19937& randomGenerator) {
	for (auto& element : array) {
		std::uniform_int_distribution<int> distribution(-100, 100);
		element = distribution(randomGenerator);
	}
}

// Вывод массива на экран
void printArray(const std::vector<int>& array) {
	std::cout << "Array = { ";
	for (const auto& element : array) {
		std::cout << element << " ";
	}
	std::cout << "}";
}

// Поиск максимального значения в массиве
int findMax(const std::vector<int>& array, const size_t& arraySize, const bool& parallelMode = false) {
	int maximum = array.at(0);

#pragma omp parallel if (parallelMode) shared(array, arraySize)
	{
#pragma omp for reduction(max: maximum)
		for (int i = 0; i < arraySize; i++) {
			if (maximum < array.at(i)) {
				maximum = array.at(i);
			}
		}
	}
	return maximum;
}

// Поиск минимального значения в массиве
int findMin(const std::vector<int>& array, const size_t& arraySize, const bool& parallelMode = false) {
	int minimum = array.at(0);

#pragma omp parallel if (parallelMode) shared(array, arraySize)
	{
#pragma omp for reduction(min: minimum)
		for (int i = 0; i < arraySize; i++) {
			if (minimum > array.at(i)) {
				minimum = array.at(i);
			}
		}
	}
	return minimum;
}

int main() {
	system("chcp 1251 > nul");
	std::mt19937 randomGenerator;
	std::random_device device;
	randomGenerator.seed(device());

	size_t arraySize;
	std::cout << "Введите размер массива: ";
	std::cin >> arraySize;
	std::cout << "Инициализация массива..." << std::endl;
	std::vector<int> array(arraySize);

	std::cout << "Заполнение массива..." << std::endl;
	randomizeArray(array, randomGenerator);
	if (arraySize <= 100) {
		printArray(array);
	}

	std::cout << "\n------------------------------------------------------------" << std::endl;
	double start = omp_get_wtime();
	int result = findMax(array, arraySize);
	double finish = omp_get_wtime();
	std::cout << "Максимальное значение в массиве (Последовательный алгоритм): " << result << std::endl <<
		"Затраченное время: " << finish - start << std::endl;
	
	std::cout << "\n------------------------------------------------------------" << std::endl;
	start = omp_get_wtime();
	result = findMin(array, arraySize);
	finish = omp_get_wtime();
	std::cout << "Минимальное значение в массиве (Последовательный алгоритм): " << result << std::endl <<
		"Затраченное время: " << finish - start << std::endl;

	std::cout << "\n------------------------------------------------------------" << std::endl;
	start = omp_get_wtime();
	result = findMax(array, arraySize, true);
	finish = omp_get_wtime();
	std::cout << "Максимальное значение в массиве (Параллельный алгоритм): " << result << std::endl <<
		"Затраченное время: " << finish - start << std::endl;

	std::cout << "\n------------------------------------------------------------" << std::endl;
	start = omp_get_wtime();
	result = findMin(array, arraySize, true);
	finish = omp_get_wtime();
	std::cout << "Минимальное значение в массиве (Параллельный алгоритм): " << result << std::endl <<
		"Затраченное время: " << finish - start << std::endl;
	
	return 0;
}