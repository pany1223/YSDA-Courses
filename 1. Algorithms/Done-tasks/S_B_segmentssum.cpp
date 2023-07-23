// Copyright [2021] <Fedor Erin>
#include <algorithm>
#include <iostream>
#include <vector>

struct ArrayAndSegment {
    std::vector<int64_t> array;
    int64_t left;
    int64_t right;
};

class Window {
public:
    Window(const std::vector<int64_t>& values, const int start, const int end)
        : values_{values}, start_{start}, end_{end} {
    }

    int64_t operator[](const int index) const {
        if (index + start_ > end_) {
            return -1;
        }
        return values_[index + start_];
    }

    int Size() const {
        return end_ + 1 - start_;
    }

private:
    const std::vector<int64_t>& values_;
    const int start_;
    const int end_;
};

void MergeWindows(const Window first, const Window second, std::vector<int64_t>& buffer) {
    int index_a = 0, index_b = 0;
    while (index_a < first.Size() && index_b < second.Size()) {
        if (first[index_a] < second[index_b]) {
            buffer.push_back(first[index_a++]);
        } else {
            buffer.push_back(second[index_b++]);
        }
    }
    while (index_a < first.Size()) {
        buffer.push_back(first[index_a]);
        ++index_a;
    }
    while (index_b < second.Size()) {
        buffer.push_back(second[index_b]);
        ++index_b;
    }
}

std::vector<int64_t> GetPartialSums(const std::vector<int64_t>& values) {
    std::vector<int64_t> res;
    res.reserve(std::size(values) + 1);
    res.push_back(0);
    for (const auto value : values) {
        const auto sum = res.back() + value;
        res.push_back(sum);
    }
    return res;
}

int64_t GetSliceCount(std::vector<int64_t>& partial_sums, int start, int end, const int64_t from) {
    if (start == end) {
        return 0;
    }
    int middle = start + (end - start) / 2;
    const auto expanding = GetSliceCount(partial_sums, start, middle, from) +
                           GetSliceCount(partial_sums, middle + 1, end, from);
    const Window left_window{partial_sums, start, middle};
    const Window right_window{partial_sums, middle + 1, end};

    int64_t count{expanding};
    int left = 0, right = 0;
    while (left < left_window.Size() && right < right_window.Size()) {
        int64_t diff = right_window[right] - left_window[left];
        if (diff >= from) {
            count += right_window.Size() - right;
            ++left;
        } else {
            ++right;
        }
    }

    std::vector<int64_t> buffer;
    buffer.reserve(left_window.Size() + right_window.Size());
    MergeWindows(left_window, right_window, buffer);
    size_t index = 0;
    while (index < std::size(buffer)) {
        partial_sums[start + index] = buffer[index];
        ++index;
    }
    return count;
}

ArrayAndSegment ReadInput() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int element_count;
    std::cin >> element_count;
    std::vector<int64_t> result;
    result.reserve(element_count);
    for (int index = 0; index < element_count; ++index) {
        int64_t element;
        std::cin >> element;
        result.push_back(element);
    }

    int64_t from, to;
    std::cin >> from >> to;

    return {std::move(result), from, to};
}

int64_t CountSegments(const ArrayAndSegment& input) {
    std::vector<int64_t> partial_sums_first = GetPartialSums(input.array);
    std::vector<int64_t> partial_sums_second = partial_sums_first;

    int64_t from_left = GetSliceCount(partial_sums_first, 0,
                                      static_cast<int>(partial_sums_first.size() - 1), input.left);
    int64_t from_right = GetSliceCount(
        partial_sums_second, 0, static_cast<int>(partial_sums_second.size() - 1), input.right + 1);
    return from_left - from_right;
}

int main() {
    auto input = ReadInput();
    std::cout << CountSegments(input) << "\n";
    return 0;
}
