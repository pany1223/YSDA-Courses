// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <vector>
#include <optional>
#include <functional>
#include <deque>
#include <algorithm>

std::tuple<std::vector<uint32_t>, std::vector<char>, int> ReadInput() {
    int n_value;  // чисел массива (1, 100000), числа до 10е9
    int m_value;  // кол-во символов R/L (1, 100000)
    int k_value;  // номер порядковой статистики (1, 100000)
    std::cin >> n_value >> m_value >> k_value;
    std::vector<uint32_t> values;
    values.reserve(n_value);
    for (int i = 0; i < n_value; ++i) {
        uint32_t element;
        std::cin >> element;
        values.push_back(element);
    }
    std::vector<char> symbols;
    symbols.reserve(m_value);
    for (int i = 0; i < m_value; ++i) {
        char symbol;
        std::cin >> symbol;
        symbols.push_back(symbol);
    }
    return std::make_tuple(std::move(values), std::move(symbols), k_value);
}

template <typename ValueType, typename ComparatorType = std::less<>>
class Heap {
public:
    Heap(int capacity = 1, ComparatorType comparator = ComparatorType{}) : comparator_{comparator} {
        values_.reserve(capacity);
    }
    int Size() const {
        return static_cast<int>(std::size(values_));
    }

    std::optional<const ValueType*> Top() {
        if (std::empty(values_)) {
            return std::nullopt;
        }
        return &values_[0];
    }

    int ParentIndex(const int index) const {
        return (index - 1) / 2;
    }

    void Push(ValueType value) {
        values_.push_back(std::move(value));
        SiftUp();
    }

    std::optional<ValueType> Pop() {
        if (std::empty(values_)) {
            return std::nullopt;
        }
        auto value = std::move(values_[0]);
        values_[0] = std::move(values_.back());
        values_.pop_back();
        SiftDown();
        return value;
    }

    std::optional<ValueType> SwapTop(ValueType number) {
        if (std::empty(values_)) {
            Push(std::move(number));
            return std::nullopt;
        }
        auto value = std::move(values_[0]);
        values_[0] = std::move(number);
        SiftDown();
        return std::move(value);
    }

    void DeleteValue(ValueType* pointer) {
        if (std::empty(values_)) {
            throw std::runtime_error{"Empty heap"};
        }
        const int location = pointer - &values_[0];
        if ((location > static_cast<int>(std::size(values_)) - 1) || (location < 0)) {
            throw std::runtime_error{"No value"};
        }
        values_[location] = std::move(values_.back());
        values_.pop_back();
        SiftDown(location);
        SiftUp(location);
    }

    std::optional<ValueType> PushAndAdjustSize(ValueType number, const int threshold) {
        if (Size() >= threshold) {
            if (comparator_(number, **Top())) {
                return std::move(number);
            } else {
                return SwapTop(std::move(number));
            }
        } else {
            Push(std::move(number));
            return std::nullopt;
        }
    }

private:
    void SiftUp(int location = -1) {
        if (location == -1) {
            location = static_cast<int>(std::size(values_)) - 1;
        }
        for (int index = location; index > 0;) {
            const auto parent_index = ParentIndex(index);
            if (comparator_(values_[index], values_[parent_index])) {
                std::swap(values_[index], values_[parent_index]);
                index = parent_index;
            } else {
                break;
            }
        }
    }

    void SiftDown(const int begin_index = 0) {
        for (int index = begin_index; index * 2 + 1 < static_cast<int>(std::size(values_));) {
            int min_index = index;
            if (comparator_(values_[index * 2 + 1], values_[index])) {
                min_index = index * 2 + 1;
            }
            if (index * 2 + 2 < static_cast<int>(std::size(values_)) &&
                comparator_(values_[index * 2 + 2], values_[min_index])) {
                min_index = index * 2 + 2;
            }
            if (min_index == index) {
                break;
            }
            std::swap(values_[index], values_[min_index]);
            index = min_index;
        }
    }

    std::vector<ValueType> values_;
    ComparatorType comparator_;
};

class KStatisticSearchOverWindow {
    struct DequeValue;
    struct HeapValue {
        DequeValue* index_seen = nullptr;
        uint32_t value = 0;

        struct Comparator {
            explicit Comparator(const bool bigger) : bigger{bigger} {
            }
            bool operator()(const HeapValue& first, const HeapValue& second) const {
                if (bigger) {
                    return first.value > second.value;
                } else {
                    return first.value < second.value;
                }
            }
            bool bigger{};
        };

        HeapValue(DequeValue* deque_value, uint32_t value) : index_seen{deque_value}, value{value} {
        }

        HeapValue& operator=(HeapValue&& another);
        HeapValue(HeapValue&& other) {
            *this = std::move(other);
        }

        void PushIntoHeap(Heap<HeapValue, Comparator>& heap) {
            index_seen->heap = &heap;
            heap.Push(std::move(*this));
        }
    };

    struct DequeValue {
        Heap<HeapValue, HeapValue::Comparator>* heap;
        HeapValue* value;
    };

public:
    explicit KStatisticSearchOverWindow(const int capacity, const int k_stat)
        : minimum_heap_{k_stat, HeapValue::Comparator{true}},
          kept_heap_{capacity, HeapValue::Comparator{false}},
          k_stat_{k_stat} {
    }

    std::optional<uint32_t> Push(const uint32_t value) {
        values_seen_.push_back({&minimum_heap_, nullptr});
        HeapValue value_to_push{&values_seen_.back(), value};

        if (auto number = minimum_heap_.PushAndAdjustSize(std::move(value_to_push), k_stat_);
            number) {
            number->PushIntoHeap(kept_heap_);

            return (*minimum_heap_.Top())->value;
        }

        if (minimum_heap_.Size() == k_stat_) {
            return (*minimum_heap_.Top())->value;
        }
        return std::nullopt;
    }

    std::optional<uint32_t> Pop() {
        const auto front_value = values_seen_.front();
        values_seen_.pop_front();
        front_value.heap->DeleteValue(front_value.value);
        if (front_value.heap == &kept_heap_) {
            return (*minimum_heap_.Top())->value;
        }
        if (kept_heap_.Size() == 0) {
            return std::nullopt;
        }
        kept_heap_.Pop()->PushIntoHeap(minimum_heap_);
        return (*minimum_heap_.Top())->value;
    }

private:
    Heap<HeapValue, HeapValue::Comparator> minimum_heap_;
    Heap<HeapValue, HeapValue::Comparator> kept_heap_;
    std::deque<DequeValue> values_seen_;
    const int k_stat_;
};

KStatisticSearchOverWindow::HeapValue& KStatisticSearchOverWindow::HeapValue::operator=(
    HeapValue&& another) {
    index_seen = another.index_seen;
    value = another.value;
    index_seen->value = this;
    return *this;
}

void PrintResult(const std::optional<uint32_t> result) {
    if (result) {
        std::cout << static_cast<int>(*result) << "\n";
    } else {
        std::cout << -1 << "\n";
    }
}

void FindAndPrintKStatistics(const std::vector<uint32_t>& values, std::vector<char> symbols,
                             int k_value) {
    KStatisticSearchOverWindow search_k_stat{static_cast<int>(values.size()), k_value};
    search_k_stat.Push(values[0]);
    int index{1};

    for (const auto symbol : symbols) {
        if (symbol == 'R') {
            PrintResult(search_k_stat.Push(values[index++]));
        } else {
            PrintResult(search_k_stat.Pop());
        }
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::vector<uint32_t> values;
    std::vector<char> symbols;
    int k_value;

    std::tie(values, symbols, k_value) = ReadInput();
    FindAndPrintKStatistics(values, symbols, k_value);

    return 0;
}
