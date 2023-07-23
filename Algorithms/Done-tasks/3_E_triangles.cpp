// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <random>
#include <memory>

int max_int = std::numeric_limits<int32_t>::max();

struct Triangle {
    int a_length;
    int b_length;
    int c_length;
};

bool operator==(const std::shared_ptr<Triangle>& first, const Triangle& second) {
    return first->a_length == second.a_length && first->b_length == second.b_length &&
           first->c_length == second.c_length;
}

class TriangleHash {
public:
    explicit TriangleHash(int a_value, int b_value = max_int) : a_val_(a_value), b_val_(b_value) {
    }

    int HashFunction(const Triangle& triangle) {
        return ((static_cast<int64_t>(triangle.a_length) * a_val_) +
                (static_cast<int64_t>(triangle.b_length) + b_val_) +
                (static_cast<int64_t>(triangle.c_length) * a_val_ + b_val_)) %
               b_val_;
    }

private:
    const int a_val_;
    const int b_val_;
};

class TriangleHashTable {
public:
    explicit TriangleHashTable(size_t size) : generator_(), classes_(size) {
        std::uniform_int_distribution<int> random_number(1, max_int - 1);
        hash_func_ = std::make_shared<TriangleHash>(random_number(generator_));
    }

    bool IsItNewTriangleClass(std::vector<std::shared_ptr<Triangle>> class_hash,
                              Triangle triangle) {
        return (class_hash.empty() ||
                std::find(class_hash.begin(), class_hash.end(), triangle) == class_hash.end());
    }

    bool SucceedInAddingNewTriangleClass(const Triangle& triangle) {
        int hash = hash_func_->HashFunction(triangle) % classes_.size();
        if (IsItNewTriangleClass(classes_[hash], triangle)) {
            classes_[hash].push_back(std::make_shared<Triangle>(triangle));
            return true;
        } else {
            return false;
        }
    }

private:
    std::mt19937 generator_{};
    std::shared_ptr<TriangleHash> hash_func_;
    std::vector<std::vector<std::shared_ptr<Triangle>>> classes_;
};

template <typename Comparable>
void InplaceSortThreeElements(Comparable& elem_a, Comparable& elem_b, Comparable& elem_c) {
    if (elem_a > elem_c) {
        std::swap(elem_a, elem_c);
    }
    if (elem_a > elem_b) {
        std::swap(elem_a, elem_b);
    }
    if (elem_b > elem_c) {
        std::swap(elem_b, elem_c);
    }
}

std::vector<Triangle> ReadInput() {
    // n (1 ≤ n ≤ 1 000 000) — кол-во треугольников
    // a, b, c (1 ≤ a, b, c ≤ 1 000 000 000) - длины сторон треугольников
    int n_value, a_length, b_length, c_length;
    std::vector<Triangle> triangles;
    Triangle triangle;

    std::cin >> n_value;
    triangles.reserve(n_value);
    for (int i = 0; i < n_value; ++i) {
        std::cin >> a_length >> b_length >> c_length;
        triangle.a_length = a_length;
        triangle.b_length = b_length;
        triangle.c_length = c_length;
        triangles.push_back(triangle);
    }
    return triangles;
}

int FindNumberOfTrianglesSimilarityClasses(const std::vector<Triangle>& triangles) {
    int number_of_classes = 0;
    TriangleHashTable classes(triangles.size());

    for (const auto& triangle : triangles) {
        std::vector edges{triangle.a_length, triangle.b_length, triangle.c_length};
        InplaceSortThreeElements(edges[0], edges[1], edges[2]);
        int gcd = std::gcd(std::gcd(edges[0], edges[1]), edges[2]);

        Triangle new_triangle{edges[0] / gcd, edges[1] / gcd, edges[2] / gcd};
        if (classes.SucceedInAddingNewTriangleClass(new_triangle)) {
            ++number_of_classes;
        }
    }
    return number_of_classes;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::vector<Triangle> triangles = ReadInput();
    std::cout << FindNumberOfTrianglesSimilarityClasses(triangles);

    return 0;
}
