// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <vector>
#include <algorithm>
#include <tuple>

class DSU {
public:
    explicit DSU(size_t size) : parent_(size), rank_(size) {
        for (size_t idx = 0; idx < size; ++idx) {
            MakeSet(idx);
        }
    }

    int FindSet(int vertex) {
        if (vertex == parent_[vertex]) {
            return vertex;
        }
        return parent_[vertex] = FindSet(parent_[vertex]);
    }

    void UniteSets(int first, int second) {
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

private:
    void MakeSet(int vertex) {
        parent_[vertex] = vertex;
        rank_[vertex] = 0;
    }

private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

struct Edge {
    int64_t weight;
    int first_vertex;
    int second_vertex;
};

class Graph {
public:
    Graph(size_t edge_n, std::istream &input = std::cin) : edges_(edge_n) {
        for (auto &edge : edges_) {
            input >> edge.first_vertex;
            --edge.first_vertex;
            input >> edge.second_vertex;
            --edge.second_vertex;
            input >> edge.weight;
        }

        std::sort(edges_.begin(), edges_.end(), [](const Edge &lhs, const Edge &rhs) {
            return std::tie(lhs.weight, lhs.first_vertex, lhs.second_vertex) <
                   std::tie(rhs.weight, rhs.first_vertex, rhs.second_vertex);
        });
    }

    const Edge &GetEdge(size_t idx) const {
        return edges_[idx];
    }

    size_t GetGraphSize() const {
        return edges_.size();
    }

private:
    std::vector<Edge> edges_;
};

int64_t FindMinSpanningTree(DSU &dsu, const Graph &graph) {
    int64_t max_edge_weight = 0;

    int64_t cost = 0;
    for (size_t idx = 0; idx < graph.GetGraphSize(); ++idx) {
        const auto &edge = graph.GetEdge(idx);
        if (dsu.FindSet(edge.first_vertex) != dsu.FindSet(edge.second_vertex)) {
            cost += edge.weight;
            max_edge_weight = std::max(max_edge_weight, edge.weight);
            dsu.UniteSets(edge.first_vertex, edge.second_vertex);
        }
    }

    return max_edge_weight;
}

int main() {
    size_t vertex_n, edge_n;
    std::cin >> vertex_n;
    std::cin >> edge_n;

    DSU dsu(vertex_n);
    Graph graph(edge_n);

    std::cout << FindMinSpanningTree(dsu, graph) << "\n";

    return 0;
}
