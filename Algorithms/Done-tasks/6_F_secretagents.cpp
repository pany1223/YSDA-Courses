// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <vector>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <stdio.h>

struct Coordinate {
    double first;
    double second;
};


double Distance(const Coordinate& lhs, const Coordinate& rhs) {
    return std::sqrt((lhs.first - rhs.first) * (lhs.first - rhs.first) +
                     (lhs.second - rhs.second) * (lhs.second - rhs.second));
}

struct Edge {
    double weight;
    size_t first_vertex;
    size_t second_vertex;
};

const int64_t kMaxCoordinate = 1'000'000'000;
const int64_t kMinCoordinate = -1'000'000'000;
const double kMinRadius = 0.0;
const double kMaxRadius =
        Distance({kMinCoordinate, kMinCoordinate}, {kMaxCoordinate, kMaxCoordinate});

class Graph {
public:
    explicit Graph(const std::vector<Coordinate>& agents) {
        edges_.reserve(agents.size() * agents.size());
        for (size_t idx = 0; idx < agents.size(); ++idx) {
            for (size_t jdx = idx + 1; jdx < agents.size(); ++jdx) {
                edges_.push_back({Distance(agents[idx], agents[jdx]), idx, jdx});
            }
        }

        std::sort(edges_.begin(), edges_.end(), [](const Edge& lhs, const Edge& rhs) {
            return std::tie(lhs.weight, lhs.first_vertex, lhs.second_vertex) <
                   std::tie(rhs.weight, rhs.first_vertex, rhs.second_vertex);
        });
    }

    const Edge& GetEdge(size_t idx) const {
        return edges_[idx];
    }

    size_t GetGraphSize() const {
        return edges_.size();
    }

private:
    std::vector<Edge> edges_;
};

class DSU {
public:
    explicit DSU(size_t size) : parent_(size), rank_(size) {
        for (size_t idx = 0; idx < size; ++idx) {
            MakeSet(idx);
        }
    }

    void UniteSets(size_t first, size_t second) {
        first = FindSet(first);
        second = FindSet(second);

        if (first != second) {
            if (rank_[first] < rank_[second]) {
                std::swap(first, second);
            }
            parent_[first] = parent_[second];
            if (rank_[first] == rank_[second]) {
                ++rank_[first];
            }
        }
    }

    bool CheckConnectivity() {
        size_t parent = FindSet(0);
        for (size_t vertex: parent_) {
            if (parent != FindSet(vertex)) {
                return false;
            }
        }
        return true;
    }

private:
    void MakeSet(size_t vertex) {
        parent_[vertex] = vertex;
        rank_[vertex] = 0;
    }

    size_t FindSet(size_t vertex) {
        if (vertex == parent_[vertex]) {
            return vertex;
        }
        return parent_[vertex] = FindSet(parent_[vertex]);
    }

private:
    std::vector<size_t> parent_;
    std::vector<size_t> rank_;
};

std::vector<Coordinate> ReadAgents(size_t agents_num) {
    std::vector<Coordinate> result(agents_num);
    for (auto& agent : result) {
        std::cin >> agent.first;
        std::cin >> agent.second;
    }
    return result;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    size_t agents_num;
    std::cin >> agents_num;

    auto agents = ReadAgents(agents_num);
    Graph graph(agents);

    double left = kMinRadius;
    double right = kMaxRadius;

    while (right - left >= 0.00001) {
        double mid = (right + left) / 2;

        DSU dsu(agents_num);

        for (size_t idx = 0; idx < graph.GetGraphSize(); ++idx) {
            const auto& edge = graph.GetEdge(idx);
            if (edge.weight < mid) {
                dsu.UniteSets(edge.first_vertex, edge.second_vertex);
            }
        }

        if (dsu.CheckConnectivity()) {
            right = (right + left) / 2;
        } else {
            left = (right + left) / 2;
        }
    }
    printf("%.10f", (right + left) / 2.0);
    return 0;
}
