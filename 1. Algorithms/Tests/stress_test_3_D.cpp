 int FindStatisticWithSort(const int statistic_number, std::vector<uint32_t> array) {
    if (statistic_number > std::size(array)) {
        return -1;
    }
    std::sort(std::begin(array), std::end(array));
    return static_cast<int>(array[statistic_number - 1]);
 }

 std::vector<int> FindStatisticsWithSort(const Input& input) {
    std::vector<int> result;
    result.reserve(std::size(input.moves));

    int left = 0, right = 1;
    for (const auto move : input.moves) {
        switch (move) {
            case Input::MoveType::Left:
                ++left;
                break;
            case Input::MoveType::Right:
                ++right;
                break;
        }
        result.push_back(FindStatisticWithSort(
            input.number, std::vector<uint32_t>(std::begin(input.array) + left,
                                                std::begin(input.array) + right)));
    }

    return result;
 }

 bool CompareVectors(const std::vector<int>& v1, const std::vector<int>& v2) {
    if (std::size(v1) != std::size(v2)) {
        std::exit(-1435);
    }
    bool differ = false;
    for (int i = 0; i < std::size(v1); ++i) {
        if (v1[i] != v2[i]) {
            differ = true;
            std::cout << "Position " << i << " differ: "
                      << "v1: " << v1[i] << ", v2: " << v2[i] << "\n";
        }
    }

    if (!differ) {
        std::cout << "vectors equal\n";
    }

    return !differ;
 }

 template <typename Container>
 void PrintContainer(const Container& container) {
    for (const auto& el : container) {
        std::cout << el << " ";
    }
    std::cout << "\n";
 }

 bool CheckCorrect(const std::vector<char>& array) {
    int rs = 1, ls = 1;
    for (const auto element : array) {
        if (element == 'L') {
            ++ls;
        } else {
            ++rs;
        }

        if (ls > rs) {
            return false;
        }
    }

    return true;
 }

 void IterateCorrectPermutations(Input& input) {
    int n = (int)std::size(input.array);
    input.moves.resize(2 * n - 2);

    std::vector<char> res(2 * n - 2, 'L');
    for (int i = 0; i < n - 1; ++i) {
        res[i] = 'R';
    }

    const auto patch_input = [&]() {
        for (int i = 0; i < std::size(res); ++i) {
            switch (res[i]) {
                case 'L':
                    input.moves[i] = Input::MoveType::Left;
                    break;
                case 'R':
                    input.moves[i] = Input::MoveType::Right;
                    break;
            }
        }
    };

    do {
        if (!CheckCorrect(res)) {
            continue;
        }

        patch_input();

        const auto statistics = FindStatistics(input);
        const auto statistics1 = FindStatisticsWithSort(input);
        if (!CompareVectors(statistics, statistics1)) {
            PrintContainer(res);
            PrintContainer(input.array);
            std::cout << "k: " << input.number << "\n";
            std::abort();
        }

    } while (std::next_permutation(std::begin(res), std::end(res), std::greater<>{}) &&
             res[0] != 'L');
 }

 std::vector<uint32_t> GenerateArray(const int n) {
    std::mt19937 generator{};
    std::uniform_int_distribution<uint32_t> val_distribution{0,
                                                             1'000'000'000
    };

    std::vector<uint32_t> result;
    result.reserve(n);
    for (int i = 0; i < n; ++i) {
        result.push_back(val_distribution(generator));
    }

    return result;
 }

 void RunStress() {
    Input input;
    for (int i = 2; i <= 20; ++i) {
        auto array = GenerateArray(i);
        input.array = std::move(array);
        for (int k = 1; k <= i; ++k) {
            input.number = k;
            IterateCorrectPermutations(input);
        }
    }
 }
