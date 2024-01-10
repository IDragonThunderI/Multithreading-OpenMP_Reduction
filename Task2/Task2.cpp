#include <iostream>
#include <random>
#include <iomanip>
#include <numbers>
#include <functional>
#include <string>
#include <omp.h>

// ГПСЧ в диапазоне от 0 до 1
double randomNumber(std::mt19937& randomGenerator) {
	std::uniform_real_distribution<double> distribution(0, 1);
	return distribution(randomGenerator);
}

// Метод Монте-Карло
long double MonteCarloPiCalculation(const int& points, std::mt19937& randomGenerator, const bool& parallelMode = false) {
	unsigned pointsInsideCircle = 0;
	double x, y;
#pragma omp parallel if (parallelMode)
	{
#pragma omp for reduction(+: pointsInsideCircle)
		for (int i = 0; i < points; i++) {
			x = randomNumber(randomGenerator);
			y = randomNumber(randomGenerator);
			if (x * x + y * y <= 1.0) {
				pointsInsideCircle++;
			}
		}
	}
	return 4.0 * pointsInsideCircle / points;
}

// Метод Симпсона
long double integrateSimpson(const double& a, const double& b, int& n, const std::function<double (const double&)> function, const bool& parallelMode = false) {
	if (n == 0) {
		return (b - a) / 6.0 * (function(a) + 4.0 * function((a + b) / 2.0) + function(b));
	}
	if (n & 1) {
		n++;
	}
	double h = (b - a) / n;
	double result = function(a) + 4.0 * function(a + h) + function(b);

#pragma omp parallel if (parallelMode)
	{
#pragma omp for reduction(+: result)
		for (int i = 1; i < n / 2; i++) {
			result += 2.0 * function(a + (2.0 * i) * h) + 4.0 * function(a + (2.0 * i + 1.0) * h);
		}
	}
	return result * h / 3.0;
}

int main() {
	system("chcp 1251 > nul");
	std::mt19937 randomGenerator;
	std::random_device device;
	randomGenerator.seed(device());

	constexpr long double PI{ std::numbers::pi_v<long double> };
	std::cout << "Точное значение числа Pi: " << std::setprecision(19) << PI << std::endl;
	std::cout << "---------------------------------\n\n";

	int points;
	std::cout << "Введите количество случайных точек: ";
	std::cin >> points;

	std::cout << "---------------------------------\n";
	std::cout << "Вычисление...\n";
	long double result = MonteCarloPiCalculation(points, randomGenerator, true);
	std::cout << "Значение чиcла Pi (метод Монте-Карло): " << result << "\n\n";

	int n;
	std::cout << "Введите количество разбиений интегрального отрезка: ";
	std::cin >> n;

	std::cout << "---------------------------------\n";
	std::cout << "Вычисление...\n";
	result = integrateSimpson(0, 1, n, [](const double& x) -> double { return 4 / (1 + x * x); }, true);
	std::cout << "Значение числа Pi (по интегралу): " << result << std::endl;

	return 0;
}