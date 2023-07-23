// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

const double MAX_RADIUS = 1000 * sqrt(2);  //  диагональ квадрата 1000х1000
const double PRECISION = 0.000001;  // точность нахождения радиуса

// точка на оси Х с флагом "конец ли это отрезка"
struct Dot {
    double dot_coord;
    bool end_flag;
};

// для сравнения структур точек
bool dots_comparator(const Dot& dot_one, const Dot& dot_two) {
    if (dot_one.dot_coord == dot_two.dot_coord) {
        return dot_one.end_flag < dot_two.end_flag;
    } else {
        return dot_one.dot_coord < dot_two.dot_coord;
    }
}

// границы на оси Х для каждой точки, в которых должны быть
// центры окружностей с заданным радиусом, чтобы накрыть точки
std::vector<std::vector<double>>
    x_ranges(const std::vector<std::vector<double>>& dots,
             double radius) {
    std::vector<int> extremums_indices;
    std::vector<std::vector<double>> ranges;
    for (size_t i = 0; i < dots.size(); ++i) {
        double under_root = pow(radius, 2) - pow(dots[i][1], 2);
        if (under_root >= 0) {
            double root = sqrt(under_root);
            ranges.push_back({dots[i][0] - root, dots[i][0] + root});
        }
    }
    return ranges;
}

// проверка наличия общего интервала для хотя бы К отрезков
bool have_k_intersections(std::vector<std::vector<double>> x_ranges,
                          int k_intersections) {
    std::vector<Dot> x_ranges_flatten(x_ranges.size() * 2);
    // складываем все отрезки в массив точек с флагами начала/конца отрезка
    for (size_t i = 0; i < x_ranges.size(); ++i) {
        x_ranges_flatten[i*2] = Dot{x_ranges[i][0], false};
        x_ranges_flatten[i*2+1] = Dot{x_ranges[i][1], true};
    }
    // сортировка отрезков
    std::sort(x_ranges_flatten.begin(),
              x_ranges_flatten.end(),
              dots_comparator);
    // ищем момент, когда вошли в k отрезов (и еще не вышли)
    int k_value = 0;
    for (size_t j = 0; j < x_ranges_flatten.size(); ++j) {
        x_ranges_flatten[j].end_flag ? --k_value : ++k_value;
        if (k_value == k_intersections) {
            return true;
        }
    }
    return false;
}

// бин поиск минимального радиуса окружности в интервале (0, MAX_RADIUS)
// на оси OХ с покрытием k_count точек из dots
double find_radius(const std::vector<std::vector<double>>& dots, int k_count) {
    double left = 0.0;
    double right = MAX_RADIUS;

    while (right - left > PRECISION) {
        double middle = left + (right - left) / 2;
        if (have_k_intersections(x_ranges(dots, middle), k_count)) {
            right = middle;
        } else {
            left = middle;
        }
    }
    return right;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // 1 <= k <= n <= 10000
    int n_dots, k_count;
    std::cin >> n_dots >> k_count;
    std::vector<std::vector<double>> coordinates(n_dots,
                                                 std::vector<double> (2, 0));

    // координаты n точек, по модулю не превосходят 1000
    for (int i = 0; i < n_dots; ++i) {
        std::cin >> coordinates[i][0] >> coordinates[i][1];
    }
    double result = find_radius(coordinates, k_count);
    std::cout << std::fixed;
    std::cout.precision(6);
    std::cout << result << "\n";
    return 0;
}
