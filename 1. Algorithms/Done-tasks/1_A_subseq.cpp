// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <algorithm>
#include <vector>

std::vector<int> find_extremums(const std::vector<int>& array) {
    int size = static_cast<int>(array.size());
    std::vector<int> extremums_indices;
    // запишем индексы точек экстремумов
    // начало и конец - экстремумы по умолчанию
    // индексы минимумов запишем со знаком "-"
    extremums_indices.push_back(0);
    for (int i = 1; i < size-1; ++i) {
        if (array[i-1] < array[i] && array[i] > array[i+1]) {
            extremums_indices.push_back(i);
        } else if (array[i-1] > array[i] && array[i] < array[i+1]) {
            extremums_indices.push_back(-i);
        }
    }
    // последний элемент либо максимум, либо минимум
    if (array[size-1] > array[abs(extremums_indices.back())]) {
        extremums_indices.push_back(size-1);
    } else {
        extremums_indices.push_back(1-size);
    }
    return extremums_indices;
}

// LAS = Longest Alternating Subsequence
std::vector<int> LAS(const std::vector<int>& array) {
    if (array.size() <= 2) {
        return array;
    }
    int element;
    std::vector<int> subseq;
    std::vector<int> extremums_indices = find_extremums(array);
    // первый кладем всегда
    subseq.push_back(array[0]);
    size_t max_index = extremums_indices.size() - 1;
    for (size_t index = 0; index < max_index; ++index) {
        // смотрим 2 смежных экстремума (abs-ом отбрасываем минус у минимума)
        size_t left = abs(extremums_indices[index]);
        size_t middle = abs(extremums_indices[index+1]);
        // индекс элемента, который смотрим сейчас
        size_t current = left + 1;
        // если справа минимум
        if (extremums_indices[index+1] < 0) {
            // если после минимума есть максимум
            element = (index + 1 < max_index) ?
                std::min(array[abs(extremums_indices[index+2])],
                         subseq.back()) :
                std::min(array[left],
                         subseq.back());
            // ищем элемент с наименьшим индексом
            while (current <= middle && array[current] >= element) {
                current++;
            }
        // аналогично с максимумом
        } else {
            element = (index + 1 < max_index) ?
                std::max(array[abs(extremums_indices[index+2])],
                         subseq.back()) :
                std::max(array[left],
                         subseq.back());
            while (current <= middle && array[current] <= element) {
                current++;
            }
        }
        subseq.push_back(array[current]);
    }
    return subseq;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int size;
    std::cin >> size;
    std::vector<int> array;
    int current;
    std::cin >> current;
    array.push_back(current);
    int previous = current;

    for (int i = 1; i < size; ++i) {
        // сразу очищаем от смежных равных элементов
        std::cin >> current;
        if (current != previous) {
            array.push_back(current);
        }
        previous = current;
    }
    std::vector<int> answer = LAS(array);
    for (int value : answer) {
        std::cout << value << ' ';
    }
    std::cout << "\n";
    return 0;
}
