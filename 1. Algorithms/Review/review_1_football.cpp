// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <vector>
#include <tuple>

namespace {
struct Player {
    int index;
    uint64_t score;
};

struct TeamOfPlayersWithAscendingSortedScores {
    std::vector<Player>::iterator team_start_index;
    std::vector<Player>::iterator team_end_index;
};

bool IndexComparator(const Player& first, const Player& second) {
    return first.index < second.index;
}

bool ScoreComparator(const Player& first, const Player& second) {
    return first.score < second.score;
}

template <typename Iter, typename Comparator>
std::vector<typename Iter::value_type> Merge(Iter begin, Iter middle, Iter end,
                                             Comparator comparator) {
    std::vector<typename Iter::value_type> result_iter;
    result_iter.reserve(std::distance(begin, end));
    Iter iter_left{begin};
    Iter iter_right{middle};
    Iter iter_middle{middle};
    Iter iter_end{end};

    while (iter_left != iter_middle && iter_right != iter_end) {
        if (comparator(*iter_left, *iter_right)) {
            result_iter.push_back(*iter_left++);
        } else {
            result_iter.push_back(*iter_right++);
        }
    }
    result_iter.insert(result_iter.end(), iter_left, iter_middle);
    result_iter.insert(result_iter.end(), iter_right, iter_end);

    return result_iter;
}

template <typename Iter, typename Comparator>
void MergeSort(Iter begin, Iter end, Comparator comparator) {
    auto size = std::distance(begin, end);
    if (size < 2) {
        return;
    }
    auto middle = std::next(begin, size / 2);
    MergeSort(begin, middle, comparator);
    MergeSort(middle, end, comparator);

    auto result_iter = Merge(begin, middle, end, comparator);
    std::move(result_iter.begin(), result_iter.end(), begin);
}

uint64_t GetTotalTeamScore(std::vector<Player>& players_scores) {
    uint64_t sum_scores = 0;
    for (const auto val : players_scores) {
        sum_scores += val.score;
    }
    return sum_scores;
}

bool DoesLastPlayerSuit(TeamOfPlayersWithAscendingSortedScores& current_team) {
    return ((current_team.team_end_index < current_team.team_start_index + 2) ||
            (current_team.team_start_index->score + (current_team.team_start_index + 1)->score >=
             current_team.team_end_index->score));
}

}  // namespace

std::vector<Player> BuildMostEffectiveSolidaryTeam(std::vector<Player>& players_scores) {
    MergeSort(players_scores.begin(), players_scores.end(), ScoreComparator);

    uint64_t result_score = 0;
    uint64_t current_score = 0;
    TeamOfPlayersWithAscendingSortedScores result_team = {players_scores.begin(),
                                                          ++players_scores.begin()};
    TeamOfPlayersWithAscendingSortedScores current_team = {players_scores.begin(),
                                                           ++players_scores.begin()};

    while (current_team.team_end_index != players_scores.end()) {
        if (DoesLastPlayerSuit(current_team)) {
            current_score += current_team.team_end_index->score;
            ++current_team.team_end_index;
        } else {
            current_score -= current_team.team_start_index->score;
            ++current_team.team_start_index;
        }
        if (current_score > result_score) {
            result_score = current_score;
            result_team = current_team;
        }
    }

    return {result_team.team_start_index, result_team.team_end_index};
}

std::vector<Player> ReadInput() {
    int n_players;
    std::cin >> n_players;
    std::vector<Player> players_scores(n_players);

    for (int i = 0; i < n_players; ++i) {
        players_scores[i].index = i + 1;
        std::cin >> players_scores[i].score;
    }

    return players_scores;
}

void WriteOutput(std::vector<Player> players_scores) {
    uint64_t sum_scores = GetTotalTeamScore(players_scores);
    MergeSort(players_scores.begin(), players_scores.end(), IndexComparator);

    std::cout << sum_scores << '\n';
    for (const auto val : players_scores) {
        std::cout << val.index << ' ';
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::vector<Player> players_scores = ReadInput();
    std::vector<Player> result_team = BuildMostEffectiveSolidaryTeam(players_scores);
    WriteOutput(result_team);

    return 0;
}
