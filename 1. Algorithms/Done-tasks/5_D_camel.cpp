// Copyright [2021] <Fedor Erin>
#include <algorithm>
#include <iostream>
#include <stack>
#include <vector>
#include <utility>

int max_camels = 2;
int infinite = 10000 * max_camels;

enum Color { green, red };

struct Oasis {
    int from;
    int to;
    int camels;
};

struct OasisData {
    Color color = green;
    int camels = infinite;
};


class OasisesGraph {
public:
    struct Route {
        int destination_oasis;
        int camels;
    };

    explicit OasisesGraph(int n_oasises) : n_oasises(n_oasises), routes(n_oasises) {
    }

    void AddEdge(Oasis route) {
        routes[route.from].push_back({route.to, route.camels});
    }

public:
    int n_oasises;
    std::vector<std::vector<Route>> routes;
};


class Queue {
public:
    std::vector<std::stack<int>> queue;
    int index = 0;

public:
    Queue(int n_possible_camels, int from_oasis) : queue(n_possible_camels + 1) {
        queue.front().push(from_oasis);
    };

    bool Empty() {
        for (const auto& stack : queue) {
            if (!stack.empty()) {
                return false;
            }
        }
        return true;
    }

    int GetOasis() {
        while (queue[GetBucketIndex(index)].empty()) {
            ++index;
        }
        auto top = queue[GetBucketIndex(index)].top();
        queue[GetBucketIndex(index)].pop();

        return top;
    }

    void AddOasis(int camels, int oasis) {
        queue[GetBucketIndex(camels)].push(oasis);
    }

private:
    int GetBucketIndex(int value) {
        return value % (max_camels + 1);
    }
};

// Dial’s Algorithm (Optimized Dijkstra for small range weights)
void VisitAdjacentOasises(OasisesGraph& graph, int from_oasis, std::vector<OasisData>& oasis_data,
                          Queue& queue) {
    for (auto route : graph.routes[from_oasis]) {
        int oasis = route.destination_oasis;
        if (oasis_data[oasis].color == green) {
            oasis_data[oasis].camels =
                std::min(oasis_data[from_oasis].camels + route.camels, oasis_data[oasis].camels);
            queue.AddOasis(oasis_data[oasis].camels, oasis);
        }
    }
}

int FindMinCamelsPathBetweenOasises(OasisesGraph& graph, std::pair<int, int> request) {
    auto [from_oasis, to_oasis] = request;
    Queue queue(max_camels + 1, from_oasis);
    std::vector<OasisData> all_oasises(graph.n_oasises);
    all_oasises[from_oasis] = {green, 0};

    while (!queue.Empty()) {
        int oasis = queue.GetOasis();
        if (all_oasises[oasis].color == green) {
            VisitAdjacentOasises(graph, oasis, all_oasises, queue);
            all_oasises[oasis].color = red;
        }
    }
    return all_oasises[to_oasis].camels;
}

std::vector<int> HandleRequests(const std::vector<std::pair<int, int>>& requests,
                                OasisesGraph& graph) {
    std::vector<int> result;
    for (auto request : requests) {
        int answer = FindMinCamelsPathBetweenOasises(graph, request);
        result.push_back(answer < infinite ? answer : -1);
    }
    return result;
}

std::pair<OasisesGraph, std::vector<std::pair<int, int>>> ReadInput() {
    int n_oasis, m_routes;
    int a_from, b_to, c_camels;
    int k_requests;
    int x_from, y_to;

    std::cin >> n_oasis >> m_routes;
    OasisesGraph graph(n_oasis);

    for (int i = 0; i < m_routes; ++i) {
        std::cin >> a_from >> b_to >> c_camels;
        graph.AddEdge({a_from - 1, b_to - 1, c_camels});
    }

    std::cin >> k_requests;
    std::vector<std::pair<int, int>> requests;

    for (int i = 0; i < k_requests; ++i) {
        std::cin >> x_from >> y_to;
        requests.push_back({x_from - 1, y_to - 1});
    }
    return {graph, requests};
}

void WriteOutput(const std::vector<int>& camels) {
    for (auto camel : camels) {
        std::cout << camel << "\n";
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    auto [graph, requests] = ReadInput();
    auto answers = HandleRequests(requests, graph);
    WriteOutput(answers);

    return 0;
}
