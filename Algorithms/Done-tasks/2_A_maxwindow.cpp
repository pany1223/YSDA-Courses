// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <vector>
#include <tuple>
#include <deque>

std::tuple<std::vector<int>, std::vector<char>> ReadInput() {
    // n (1 ≤ n ≤ 100 000) — размер массива
    // n целых чисел от -1 000 000 000 до 1 000 000 000 — сам массив
    // m (0 ≤ m ≤ 2n - 2) — количество перемещений
    // m символов L или R, разделенных пробелами
    int n_value, m_value;
    std::vector<int> array;
    std::vector<char> commands;

    std::cin >> n_value;
    for (int i = 0; i < n_value; ++i) {
        int value;
        std::cin >> value;
        array.push_back(value);
    }

    std::cin >> m_value;
    for (int j = 0; j < m_value; ++j) {
        char letter;
        std::cin >> letter;
        commands.push_back(letter);
    }

    return std::make_tuple(array, commands);
}

void MaxInSlidingWindow(const std::vector<int>& array, const std::vector<char>& commands) {
    int l_pointer = 0;
    int r_pointer = 0;
    std::deque<int> decreasing_maximums;  // убывающие максимумы
    std::deque<int>
        deque_values_indices_in_array;  // индексы элементов убыв. макс-ов в исходном массиве

    // первый элемент всегда кладем
    decreasing_maximums.push_back(array[0]);
    deque_values_indices_in_array.push_back(0);

    for (const auto command : commands) {
        if (command == 'R') {
            r_pointer += 1;
            // удаляем с конца все, что меньше нового элемента (сохраняем убывание в
            // decreasing_maximums)
            while (!decreasing_maximums.empty() && decreasing_maximums.back() <= array[r_pointer]) {
                decreasing_maximums.pop_back();
                deque_values_indices_in_array.pop_back();
            }
            // кладем новый элемент и его индекс
            decreasing_maximums.push_back(array[r_pointer]);
            deque_values_indices_in_array.push_back(r_pointer);
        } else if (command == 'L') {
            // если левый край окна равен текущему максимуму, то удаляем его
            if (l_pointer == deque_values_indices_in_array.front()) {
                decreasing_maximums.pop_front();
                deque_values_indices_in_array.pop_front();
            }
            l_pointer += 1;
        }
        // в голове deque всегда максимум всего окна
        std::cout << decreasing_maximums.front() << " ";
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::vector<int> array;
    std::vector<char> commands;

    std::tie(array, commands) = ReadInput();
    MaxInSlidingWindow(array, commands);

    return 0;
}
