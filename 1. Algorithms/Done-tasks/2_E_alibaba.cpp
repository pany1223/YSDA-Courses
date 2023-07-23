// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <vector>
#include <algorithm>

struct Coin {
    int coordinate;
    int lifetime;
};

struct Interval {
    int begin;
    int end;
};

struct Step {
    std::vector<Interval> before;
    std::vector<Interval> now;
};

bool CoordinateComparator(const Coin& first, const Coin& second) {
    return first.coordinate < second.coordinate;
}

int ChooseMinimumTime(int& united, int& separated) {
    if (separated < united) {
        std::swap(united, separated);
    }
    if (united == -100) {
        return separated;
    } else {
        return united;
    }
}

int GetLeftTimeUnited(std::vector<Coin>& coins, Step& step, int& left_index) {
    int extra_path = coins[left_index + 1].coordinate - coins[left_index].coordinate;
    int time = step.before[left_index + 1].begin + extra_path;
    if (step.before[left_index + 1].begin == -100 || coins[left_index].lifetime < time) {
        return -100;
    } else {
        return time;
    }
}

int GetLeftTimeSeparated(std::vector<Coin>& coins, Step& step, int& left_index, int& length) {
    int extra_path = coins[left_index + length - 1].coordinate - coins[left_index].coordinate;
    int time = step.before[left_index + 1].end + extra_path;
    if (step.before[left_index + 1].end == -100 || coins[left_index].lifetime < time) {
        return -100;
    } else {
        return time;
    }
}

int GetRightTimeUnited(std::vector<Coin>& coins, Step& step, int& left_index, int& row) {
    int extra_path =
        coins[left_index + row - 1].coordinate - coins[left_index + row - 2].coordinate;
    int time = step.before[left_index].end + extra_path;
    if (step.before[left_index].end == -100 || coins[left_index + row - 1].lifetime < time) {
        return -100;
    } else {
        return time;
    }
}

int GetRightTimeSeparated(std::vector<Coin>& coins, Step& step, int& left_index, int& row) {
    int extra_path = coins[left_index + row - 1].coordinate - coins[left_index].coordinate;
    int time = step.before[left_index].begin + extra_path;
    if (step.before[left_index].begin == -100 || coins[left_index + row - 1].lifetime < time) {
        return -100;
    } else {
        return time;
    }
}

int GetMinimumTimeForCollectingAllCoins(std::vector<Coin>& coins) {
    std::sort(coins.begin(), coins.end(), CoordinateComparator);
    if (coins.size() == 1) {
        return 0;
    }
    Step step;
    step.now.resize(coins.size() - 1);
    step.before.resize(coins.size() - 1);
    for (int num_coins = 2; num_coins <= static_cast<int>(coins.size()); ++num_coins) {
        int path = coins[num_coins - 1].coordinate - coins[num_coins - 2].coordinate;
        if (path > coins[num_coins - 2].lifetime) {
            step.now[num_coins - 2].begin = -100;
        } else {
            step.now[num_coins - 2].begin = path;
        }
        if (path > coins[num_coins - 1].lifetime) {
            step.now[num_coins - 2].end = -100;
        } else {
            step.now[num_coins - 2].end = path;
        }
    }

    for (int row = 3; row < static_cast<int>(coins.size() + 1); ++row) {
        std::swap(step.before, step.now);
        int left_index = 0;
        while (left_index + row <= static_cast<int>(coins.size())) {
            int right_time_united = GetRightTimeUnited(coins, step, left_index, row);
            int right_time_separated = GetRightTimeSeparated(coins, step, left_index, row);
            int left_time_united = GetLeftTimeUnited(coins, step, left_index);
            int left_time_separated = GetLeftTimeSeparated(coins, step, left_index, row);

            step.now[left_index].begin = ChooseMinimumTime(left_time_united, left_time_separated);
            step.now[left_index].end = ChooseMinimumTime(right_time_united, right_time_separated);
            ++left_index;
        }
    }
    return ChooseMinimumTime(step.now[0].begin, step.now[0].end);
}

std::vector<Coin> ReadInput() {
    // n (1 ≤ n ≤ 1 000) — количество монет
    // В каждой строке по 2 целых числа - положение монеты и срок жизни в секундах - a, b (1 ≤ a, b
    // ≤ 1 000 000 000)
    int n_value, coord, time;
    std::cin >> n_value;
    std::vector<Coin> coins(n_value);

    for (int i = 0; i < n_value; ++i) {
        std::cin >> coord >> time;
        coins[i].coordinate = coord;
        coins[i].lifetime = time;
    }
    return coins;
}

void WriteOutput(int& minimum_time) {
    if (minimum_time != -100) {
        std::cout << minimum_time;
    } else {
        std::cout << "No solution";
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::vector<Coin> coins = ReadInput();
    int minimum_time = GetMinimumTimeForCollectingAllCoins(coins);
    WriteOutput(minimum_time);

    return 0;
}
