// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <vector>
#include <tuple>

// эффективность игрока с исходным индексом
struct Player {
    int index;
    uint64_t score;
};

// компараторы для сортировки Element как по score, так и index
bool IndexComparator(const Player& first, const Player& second) {
    return first.index < second.index;
}

bool ScoreComparator(const Player& first, const Player& second) {
    return first.score < second.score;
}

// слияние структур Element при сортировке
void Merge(std::vector<Player>& array,
           int left, int middle, int right,
           bool (*comparator)(const Player&, const Player&)) {
    // размеры левого/правого подмассива
    int left_size = middle - left + 1;
    int right_size = right - middle;
    std::vector<Player> left_subarray(left_size);
    std::vector<Player> right_subarray(right_size);
    // копируем элементы в подмассивы
    for (int i = 0; i < left_size; ++i) {
        left_subarray[i] = array[left + i];
    }
    for (int j = 0; j < right_size; ++j) {
        right_subarray[j] = array[middle + j + 1];
    }
    int index_i = 0;
    int index_j = 0;
    int index_k = left;
    // записываем элементы подмассива в исходный массив
    while (index_i < left_size &&
           index_j < right_size) {
        //
        if (comparator(left_subarray[index_i], right_subarray[index_j])) {
            array[index_k] = left_subarray[index_i];
            ++index_i;
        } else {
            array[index_k] = right_subarray[index_j];
            ++index_j;
        }
        ++index_k;
    }
    // запишем оставшиеся элементы
    while (index_i < left_size) {
        array[index_k] = left_subarray[index_i];
        ++index_i;
        ++index_k;
    }
    while (index_j < right_size) {
        array[index_k] = right_subarray[index_j];
        ++index_j;
        ++index_k;
    }
}

// сортировка слиянием
void MergeSort(std::vector<Player>& array,
               int left, int right,
               bool (*comparator)(const Player&, const Player&)) {
    if (left < right) {
        int middle = left + (right - left) / 2;
        // рекурсивно сортируем левую и правую половину
        MergeSort(array, left, middle, comparator);
        MergeSort(array, middle + 1, right, comparator);
        // мержим последним шагом две осортированные половины
        Merge(array, left, middle, right, comparator);
    }
}

std::tuple<int, std::vector<Player>> ReadInput() {
    // 1 <= n <= 100000
    int n_players;
    std::cin >> n_players;
    std::vector<Player> players_scores(n_players);

    // очки игроков и их исходные индексы
    for (int i = 0; i < n_players; ++i) {
        players_scores[i].index = i;
        std::cin >> players_scores[i].score;
    }
    return std::make_tuple(n_players, players_scores);
}

void WriteOutput(uint64_t sum, std::vector<Player> team) {
    // сортировка по индексам
    MergeSort(team, 0, static_cast<int>(team.size()) - 1, IndexComparator);
    // итоговая сумма и исходные индексы игроков
    std::cout << sum << "\n";
    for (auto val : team) {
        std::cout << val.index + 1 << ' ';
    }
    std::cout << "\n";
}

// поиск лучшей команды с наибольшей суммой в сортированном по score массиве
std::tuple<uint64_t, std::vector<Player>>
    FindBestTeam(std::vector<Player>& players_scores,
                 const int n_players) {
    // сортировка игроков по эффективности
    MergeSort(players_scores, 0, n_players - 1, ScoreComparator);

    uint64_t sum = 0;
    int min_i = 0;
    int max_j = 0;
    // поиск границ подпоследовательности [min_i, max_j)
    if (n_players <= 2) {
        for (Player elem : players_scores) {
            sum += elem.score;
        }
        max_j = n_players;
    } else {
        int min_index = 0;
        int max_index = 2;
        sum = players_scores[0].score + players_scores[1].score;
        uint64_t current_sum = sum;

        while (max_index < n_players) {
            if (players_scores[max_index].score <=
                players_scores[min_index].score +
                players_scores[min_index + 1].score) {
                current_sum += players_scores[max_index].score;
                ++max_index;
                if (current_sum > sum) {
                    sum = current_sum;
                    min_i = min_index;
                    max_j = max_index;
                }
            } else {
                current_sum -= players_scores[min_index].score;
                ++min_index;
            }
        }
    }
    // найденная команда и суммарная эффективность
    std::vector<Player> team = std::vector<Player>
        (players_scores.begin() + min_i,
         players_scores.begin() + max_j);
    return std::make_tuple(sum, team);
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // ввод
    int n_players;
    std::vector<Player> players_scores;
    std::tie(n_players, players_scores) = ReadInput();
    
    // поиск лучшей команды
    uint64_t sum;
    std::vector<Player> answer;
    std::tie(sum, answer) = FindBestTeam(players_scores, n_players);
    
    // вывод
    WriteOutput(sum, answer);

    return 0;
}
