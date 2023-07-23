// Copyright [2021] <Fedor Erin>
#include <memory>
#include <limits>
#include <optional>
#include <random>
#include <vector>

class HashFunction {
public:
    HashFunction(int a_number, int b_number, int p_number = std::numeric_limits<int>::max())
        : a_number_(a_number), b_number_(b_number), p_number_(p_number) {
    }

    uint64_t operator()(int value) const {
        return static_cast<uint64_t>(
                   (static_cast<int64_t>(value) * a_number_ % p_number_ + b_number_)) %
               p_number_;
    }

private:
    const int a_number_;
    const int b_number_;
    const int p_number_;
};

size_t SumOfSquaredBucketsSize(const std::vector<std::vector<int>>& buckets) {
    size_t sum = 0;
    for (const auto& bucket : buckets) {
        sum += bucket.size() * bucket.size();
    }
    return sum;
}

std::vector<std::vector<int>> GetSplitByBuckets(const std::shared_ptr<HashFunction>& hash_func,
                                                const std::vector<int>& values, int n_buckets) {
    int hash;
    std::vector<std::vector<int>> split_by_buckets(n_buckets);
    for (const auto& value : values) {
        hash = hash_func->operator()(value) % n_buckets;
        split_by_buckets[hash].push_back(value);
    }
    return split_by_buckets;
}

template <typename Predicate>
std::shared_ptr<HashFunction> RandomHashFunction(Predicate predicate, std::mt19937* generator) {
    std::uniform_int_distribution random_number(0, std::numeric_limits<int>::max() - 1);

    int a_number = random_number(*generator);
    int b_number = random_number(*generator);
    while (!predicate(std::make_unique<HashFunction>(a_number, b_number))) {
        a_number = random_number(*generator);
        b_number = random_number(*generator);
    }
    return std::make_shared<HashFunction>(a_number, b_number);
}

class InnerHashTable {
public:
    InnerHashTable() = default;

    void Initialize(const std::vector<int>& values, std::mt19937* generator) {
        if (values.empty()) {
            return;
        }

        std::vector<std::optional<int>> hash_table;
        auto predicate = [&values, &hash_table](const std::shared_ptr<HashFunction>& hash_func) {
            int n_buckets = values.size() * values.size();
            std::vector<std::vector<int>> buckets = GetSplitByBuckets(hash_func, values, n_buckets);
            hash_table = std::vector<std::optional<int>>(n_buckets);
            for (int hash = 0; hash < static_cast<int>(buckets.size()); ++hash) {
                if (buckets[hash].size() > 1) {
                    return false;
                } else if (!buckets[hash].empty()) {
                    hash_table[hash] = buckets[hash][0];
                }
            }
            return true;
        };

        hash_func_ = RandomHashFunction(predicate, generator);
        hash_table_ = std::move(hash_table);
    }

    bool Contains(int value) const {
        if (hash_table_.empty()) {
            return false;
        }
        int hash = hash_func_->operator()(value) % hash_table_.size();
        if (hash_table_[hash]) {
            return (hash_table_[hash].value() == value);
        } else {
            return false;
        }
    }

private:
    std::shared_ptr<HashFunction> hash_func_;
    std::vector<std::optional<int>> hash_table_;
};

class OuterHashTable {
public:
    OuterHashTable() : generator_(device_()) {
    }

    void Initialize(const std::vector<int>& values) {
        if (values.empty()) {
            return;
        }

        std::vector<std::vector<int>> buckets;
        auto predicate = [&values, &buckets](const std::shared_ptr<HashFunction> hash_func) {
            buckets = GetSplitByBuckets(hash_func, values, values.size());

            if (SumOfSquaredBucketsSize(buckets) <= 4 * values.size()) {
                return true;
            } else {
                return false;
            }
        };
        hash_func_ = RandomHashFunction(predicate, &generator_);

        std::vector<std::unique_ptr<InnerHashTable>> hash_table(values.size());
        for (int hash = 0; hash < static_cast<int>(buckets.size()); ++hash) {
            std::unique_ptr<InnerHashTable> chain = std::make_unique<InnerHashTable>();
            chain->Initialize(buckets[hash], &generator_);
            hash_table[hash] = std::move(chain);
        }
        hash_table_ = std::move(hash_table);
    }

    bool Contains(int value) const {
        if (hash_table_.empty()) {
            return false;
        }
        int hash = hash_func_->operator()(value) % hash_table_.size();
        return hash_table_[hash]->Contains(value);
    }

private:
    std::random_device device_;
    std::mt19937 generator_;
    std::shared_ptr<HashFunction> hash_func_;
    std::vector<std::unique_ptr<InnerHashTable>> hash_table_;
};

class FixedSet {
public:
    FixedSet() = default;

    void Initialize(const std::vector<int>& numbers) {
        set_ = std::make_unique<OuterHashTable>();
        set_->Initialize(numbers);
    }

    bool Contains(int number) const {
        return set_->Contains(number);
    }

private:
    std::unique_ptr<OuterHashTable> set_;
};
