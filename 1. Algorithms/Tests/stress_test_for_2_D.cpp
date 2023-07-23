//
// Created by Erin Fedor on 25.10.2021.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include<bits/stdc++.h>

uint64_t CalculateWithSort(std::vector<uint32_t>& homes) {
    std::sort(std::begin(homes), std::end(homes), std::less<>{});
//    std::cout << "sorted: ";
//    for (const auto element : homes) {
//        std::cout << element << " ";
//    }
//    std::cout << '\n';
    uint64_t result = 0;
    for (int i = 0, j = static_cast<int>(std::size(homes)) - 1; i < j; ++i, --j) {
        result += static_cast<uint64_t>(homes[j]) - static_cast<uint64_t>(homes[i]);
    }

    return result;
}

template <typename Container>
void PrintContainer(const Container& container) {
    for (const auto& el : container) {
        std::cout << el << " ";
    }
    std::cout << "\n";
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


void StressTest() {
    std::mt19937 generator{};
    std::uniform_int_distribution<uint32_t> val_distribution{0,
                                                             std::numeric_limits<uint32_t>::max()};
    std::uniform_int_distribution<size_t> size_distribution{1, 10'000'000};
    for (int i = 0; i < 50; ++i) {
        const auto size = size_distribution(generator);
        std::vector<uint32_t> homes;
        homes.reserve(size);
        for (int j = 0; j < size; ++j) {
            homes.push_back(val_distribution(generator));
        }

        const auto postman = FindMedian(homes);
        const auto answer = MinSumDistanceToAllDots(homes, postman);

        const auto calc_result = CalculateWithSort(homes);
        if (answer != calc_result) {
            std::cout << "ERROR!" << "\n";
            std::cout << "iteration: " << i << "\n";
            PrintContainer(homes);
            std::cout << "algo: " << answer << " dummy " << calc_result;
            std::cout << "\n" << "median: " << postman << "\n";
            std::abort();
        }
        std::cout << "iteration: " << i << " - OK!\n";
    }
}
int main() {
    StressTest();

}