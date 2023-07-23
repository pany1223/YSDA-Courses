// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <vector>
#include <tuple>

std::tuple<int, int, std::vector<std::vector<int>>> ReadInput() {
    // 1 ≤ m, n ≤ 1000
    // n - упорядоченных по возрастанию массивов, m - размер массивов
    // элементы по модулю <= 10^9
    int n_value, m_value;
    std::cin >> n_value >> m_value;
    std::vector<std::vector<int>> arrays(n_value, std::vector<int>(m_value));

    for (int i = 0; i < n_value; ++i) {
        for (int j = 0; j < m_value; ++j) {
            std::cin >> arrays[i][j];
        }
    }
    return std::make_tuple(n_value, m_value, arrays);
}

void WriteOutput(std::vector<int>& array, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << "\n";
}

// слияние двух массивов и запись результата в result
void Merge(int left, int right, int m_value, std::vector<int>& result) {
    // начало левого и правого массивов
    int left_start = left * m_value;
    int right_start = ((left + right) / 2 + 1) * m_value;
    // размеры левого и правого массивов
    int left_size = ((left + right) / 2 - left + 1) * m_value;
    int right_size = (right - (left + right) / 2) * m_value;

    // временное хранение левого и правого массивов
    std::vector<int> left_array(left_size), right_array(right_size);
    // сохраним элементы левого и правого массивов
    for (int i = 0; i < left_size; ++i) {
        left_array[i] = result[left_start + i];
    }
    for (int i = 0; i < right_size; ++i) {
        right_array[i] = result[right_start + i];
    }

    // индексы левого, правого и итогового массивов
    int left_index = 0;
    int right_index = 0;
    int result_index = left_start;

    // слияние двух массивов
    while (left_index < left_size && right_index < right_size) {
        if (left_array[left_index] < right_array[right_index]) {
            result[result_index] = left_array[left_index];
            ++left_index;
            ++result_index;
        } else {
            result[result_index] = right_array[right_index];
            ++right_index;
            ++result_index;
        }
    }
    // если дошли до конца правого, записываем все остального из левого
    while (left_index < left_size) {
        result[result_index] = left_array[left_index];
        ++left_index;
        ++result_index;
    }
    // если дошли до конца левого, записываем все остального из правого
    while (right_index < right_size) {
        result[result_index] = right_array[right_index];
        ++right_index;
        ++result_index;
    }
}

// рекурсивное слияние массивов
void MergeNSortedArrays(int left, int right, int m_value, std::vector<int>& result,
                        const std::vector<std::vector<int>>& arrays) {
    if (left == right) {
        // если индексы равны, берем массив и пишем в результат as is
        for (int i = 0; i < m_value; ++i) {
            result[left * m_value + i] = arrays[left][i];
        }
    } else {
        // левые и правые массивы
        MergeNSortedArrays(left, (left + right) / 2, m_value, result, arrays);
        MergeNSortedArrays((left + right) / 2 + 1, right, m_value, result, arrays);
        // слияние последних двух половин набора массивов
        Merge(left, right, m_value, result);
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n_value, m_value;
    std::vector<std::vector<int>> arrays;
    std::tie(n_value, m_value, arrays) = ReadInput();

    // слияние отсортированных последовательностей, сложность - O(mnlogn), память - O(mn)
    std::vector<int> result(n_value * m_value);
    MergeNSortedArrays(0, n_value - 1, m_value, result, arrays);

    WriteOutput(result, n_value * m_value);

    return 0;
}
