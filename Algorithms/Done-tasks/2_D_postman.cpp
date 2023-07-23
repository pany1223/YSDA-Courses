// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <vector>
#include <algorithm>

uint32_t cur = 0;  // беззнаковое 32-битное число

uint32_t NextRand24(const uint32_t& input_a, const uint32_t& input_b) {
    cur = cur * input_a + input_b;  // вычисляется с переполнениями
    return cur >> 8;                // число от 0 до 2**24-1.
}

uint32_t NextRand32(const uint32_t& input_a, const uint32_t& input_b) {
    uint32_t a_value = NextRand24(input_a, input_b);
    uint32_t b_value = NextRand24(input_a, input_b);
    return (a_value << 8) ^ b_value;  // число от 0 до 2**32-1.
}

std::vector<uint32_t> ReadInput() {
    // n (1 ≤ n ≤ 10 000 000) — кол-во точек на оси Х (размер массива)
    // a, b (1 ≤ a, b ≤ 1 000 000 000) - для генератора случайных чисел
    uint32_t n_value, a_value, b_value;
    std::vector<uint32_t> dots;

    std::cin >> n_value;
    std::cin >> a_value >> b_value;
    for (uint32_t i = 0; i < n_value; ++i) {
        uint32_t x_coord = NextRand32(a_value, b_value);
        dots.push_back(x_coord);
    }

    return dots;
}

uint32_t PickPivot(const std::vector<uint32_t>& array, const int& left, const int& right) {
    // рандомный индекс из интервала [left, right] и соответствующее значение
    int pivot_index = rand() % (right - left + 1) + left;
    return array[pivot_index];
}

int Partition(std::vector<uint32_t>& array, const int& left, const int& right,
              const uint32_t& pivot) {
    int left_index = left - 1;
    int right_index = right + 1;
    int flag = 0;
    while (flag < 1) {
        left_index += 1;
        while (array[left_index] < pivot) {
            left_index += 1;
        }
        right_index -= 1;
        while (array[right_index] > pivot) {
            right_index -= 1;
        }
        if (left_index >= right_index) {
            return right_index;
        }
        std::swap(array[left_index], array[right_index]);
    }
    return 0;
}

uint32_t QuickSelect(std::vector<uint32_t>& array, const int& left, const int& right,
                     const int& kth_number) {
    if (right - left < 1) {
        return array[right];
    } else {
        uint32_t pivot = PickPivot(array, left, right);
        int split = Partition(array, left, right, pivot);
        if (kth_number <= split) {
            return QuickSelect(array, left, split, kth_number);
        } else {
            return QuickSelect(array, split + 1, right, kth_number);
        }
    }
}

uint32_t FindMedian(std::vector<uint32_t>& array) {
    int kth_number = array.size() / 2;  // k-ая статистика с k-1 индексом
    uint32_t median = QuickSelect(array, 0, array.size() - 1, kth_number);
    return median;
}

uint64_t MinSumDistanceToAllDots(const std::vector<uint32_t>& dots, uint32_t median) {
    uint64_t sum = 0;
    for (const auto x_coord : dots) {
        if (median >= x_coord) {
            sum += median - x_coord;
        } else {
            sum += x_coord - median;
        }
    }
    return sum;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::vector<uint32_t> dots = ReadInput();
    uint32_t median = FindMedian(dots);
    uint64_t result = MinSumDistanceToAllDots(dots, median);

    std::cout << result;

    return 0;
}
