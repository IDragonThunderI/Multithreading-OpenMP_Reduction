#include <iostream>
#include <cmath>
#include <iomanip>
#include <omp.h>

// Подынтегральная функция
double function(const double& x) {
    return sin(x);
}

// Итерационный метод трапеций
double integrateTrapezoidalIterative(const double& a, const double& b, const int& n, const bool& parallelMode = false) {
    double result = (function(a) + function(b)) * 0.5;
    
    if (n == 0) {
        return result * (b - a);
    }
    double h = (b - a) / n;

#pragma omp parallel if (parallelMode)
    {
#pragma omp for reduction(+: result)
        for (int i = 1; i < n; i++) {
            result += function(a + i * h);
        }
    }
    return result * h;
}

// Метод Симпсона
double integrateSimpson(const double& a, const double& b, int& n, const bool& parallelMode = false) {
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

    double a, b;
    int n;

    std::cout << "Введите нижний предел интегрирования (a): ";
    std::cin >> a;
    std::cout << "Введите верхний предел интегрирования (b): ";
    std::cin >> b;
    std::cout << "Введите количество разбиений (n): ";
    std::cin >> n;
    std::cout << std::endl;

    std::cout << "---------------------------------" << std::endl;
    double start = omp_get_wtime();
    std::cout << std::setprecision(15) << "Итерационный метод трапеций: " << integrateTrapezoidalIterative(a, b, n) << std::endl;
    double finish = omp_get_wtime();
    std::cout << std::setprecision(5) << "Затраченное время: " << finish - start << " с." << std::endl << std::endl;

    std::cout << "---------------------------------" << std::endl;
    start = omp_get_wtime();
    std::cout << std::setprecision(15) << "Параллельный итерационный метод трапеций: " << integrateTrapezoidalIterative(a, b, n, true) << std::endl;
    finish = omp_get_wtime();
    std::cout << std::setprecision(5) << "Затраченное время: " << finish - start << " с." << std::endl << std::endl;
    
    std::cout << "---------------------------------" << std::endl;
    start = omp_get_wtime();
    std::cout << std::setprecision(15) << "Метод Симпсона: " << integrateSimpson(a, b, n) << std::endl;
    finish = omp_get_wtime();
    std::cout << std::setprecision(5) << "Затраченное время: " << finish - start << " с." << std::endl << std::endl;
    
    std::cout << "---------------------------------" << std::endl;
    start = omp_get_wtime();
    std::cout << std::setprecision(15) << "Параллельный метод Симпсона: " << integrateSimpson(a, b, n, true) << std::endl;
    finish = omp_get_wtime();
    std::cout << std::setprecision(5) << "Затраченное время: " << finish - start << " с." << std::endl << std::endl;

    return 0;
}