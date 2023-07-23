// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <algorithm>

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    int nn;  // 1 ≤ n ≤ 900, нестрого возрастающие массивы Ai
    int mm;  // 1 ≤ m ≤ 900, нестрого убывающие массивы Bj
    int ll;  // 1 ≤ l ≤ 3000, длина каждого массива
    int qq;  // запросы вида (i, j), ответ такое k, что max(Aik, Bjk) минимален
    int ii;  // 1 ≤ i ≤ n
    int jj;  // 1 ≤ j ≤ m

    std::cin >> nn >> mm >> ll;

    // A[i][j], i - индекс элемента в наборе, j - индекс набора
    // Элементы массива — целые числа от 0 до 10^5 - 1 (нумеруются с 1)
    int A[ll][nn], B[ll][mm];

    for (int index_i = 0; index_i < nn; ++index_i) {
        for (int index_j = 0; index_j < ll; ++index_j) {
            std::cin >> A[index_j][index_i];
        }
    }
    for (int index_i = 0; index_i < mm; ++index_i) {
        for (int index_j = 0; index_j < ll; ++index_j) {
            std::cin >> B[index_j][index_i];
        }
    }

    std::cin >> qq;

    // выводим q чисел от 1 до l — ответы на q запросов
    for (int rr = 0; rr < qq; ++rr) {
        std::cin >> ii >> jj;
        // случаи, когда массивы не "пересекаются"
        if (A[0][ii-1] > B[0][jj-1]) {
            std::cout << 1;
        } else if (A[ll-1][ii-1] < B[ll-1][jj-1]) {
            std::cout << ll;
        } else {
            // bin search по массивам Ai, Bj в поисках нужно точки - это точка
            // "пересечения" двух "кривых" (возрастающей и неубывающей)
            size_t left = 0, right = ll;
            while (left < right) {
                size_t middle = left + (right - left) / 2;
                if (A[middle][ii-1] < B[middle][jj-1] &&
                    std::max(A[middle][ii-1], B[middle][jj-1]) >=
                    std::max(A[middle+1][ii-1], B[middle+1][jj-1])) {
                    left = middle + 1;
                } else {
                    right = middle;
                }
            }
            std::cout << left + 1;
        }
        std::cout << "\n";
    }
    return 0;
}
