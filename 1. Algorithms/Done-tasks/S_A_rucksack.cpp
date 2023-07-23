// Copyright [2021] <Fedor Erin>
#include <algorithm>
#include <vector>
#include <iostream>

struct Group {
    int64_t weight;
    int64_t cost;
    int64_t count;
};

struct Items {
    int64_t volume;
    std::vector<Group> groups;
};

class MaxQueue {
public:
    explicit MaxQueue(const int64_t capacity = 1) {
        head_.reserve(capacity);
    }

    MaxQueue(MaxQueue&&) noexcept = default;

    void Push(int64_t number) {
        if (std::empty(tail_)) {
            int64_t value = number;
            tail_.push_back({std::move(value), std::move(number)});
            return;
        }
        int64_t max = std::max(number, tail_.back().max);
        tail_.push_back({std::move(number), std::move(max)});
    }

    int64_t Pop() {
        int size_tail = static_cast<int64_t>(std::size(tail_));
        if (std::empty(head_)) {
            head_.reserve(size_tail);
            {
                auto value = std::move(tail_.back().current);
                auto max = value;
                head_.push_back({std::move(value), std::move(max)});
            }
            int64_t index = size_tail - 2;
            while (index >= 0) {
                auto max = std::max(head_.back().max, tail_[index].current);
                head_.push_back({std::move(tail_[index].current), std::move(max)});
                --index;
            }
            tail_.clear();
        }
        auto front = std::move(head_.back());
        head_.pop_back();
        return std::move(front.current);
    }

    void Clear() {
        head_.clear();
        tail_.clear();
    }

    const int64_t& Max() {
        if (std::empty(head_)) {
            return tail_.back().max;
        } else if (std::empty(tail_)) {
            return head_.back().max;
        } else {
            return std::max(tail_.back().max, head_.back().max);
        }
    }

    int64_t Size() const noexcept {
        return static_cast<int64_t>(std::size(head_) + std::size(tail_));
    }

private:
    struct Element {
        int64_t current;
        int64_t max;
    };
    std::vector<Element> head_;
    std::vector<Element> tail_;
};

class DynamicQueue {
public:
    DynamicQueue(const int64_t diff, const int64_t capacity_max)
        : queue_{capacity_max}, capacity_max_{capacity_max}, diff_{diff}, delta_{-diff_} {
    }
    DynamicQueue(DynamicQueue&&) noexcept = default;

    int64_t Push(const int64_t number) {
        delta_ += diff_;
        queue_.Push(number - delta_);
        if (queue_.Size() > capacity_max_) {
            queue_.Pop();
        }
        return queue_.Max() + delta_;
    }

    void Clear() {
        queue_.Clear();
    }

private:
    MaxQueue queue_;
    const int64_t capacity_max_;
    const int64_t diff_;
    int64_t delta_;
};

int64_t GetMaxSumValueItems(const struct Items& items) {
    std::vector<int64_t> state_before(items.volume);
    std::vector<int64_t> state_now(items.volume);

    for (const auto group : items.groups) {
        auto queue = [&] {
            return DynamicQueue{group.cost,
                                std::min(group.count + 1, static_cast<int64_t>(100'000 + 100))};
        }();
        int64_t init_weight = 0;
        while (init_weight < group.weight && init_weight <= items.volume) {
            queue.Clear();
            int64_t weight = init_weight;
            while (weight <= items.volume) {
                if (weight == 0) {
                    queue.Push(0);
                } else {
                    state_now[weight - 1] = queue.Push(state_before[weight - 1]);
                }
                weight += group.weight;
            }
            ++init_weight;
        }
        std::swap(state_before, state_now);
    }
    return *std::max_element(std::cbegin(state_before), std::cend(state_before));
}

Items ReadInput() {
    int64_t volume, n_groups;
    std::cin >> n_groups >> volume;
    std::vector<Group> groups;
    groups.reserve(n_groups);

    for (int i = 0; i < n_groups; ++i) {
        Group group{};
        std::cin >> group.weight >> group.cost >> group.count;
        groups.push_back(group);
    }

    return {volume, std::move(groups)};
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    const auto items = ReadInput();
    std::cout << GetMaxSumValueItems(items) << "\n";
    return 0;
}
