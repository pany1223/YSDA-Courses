// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

class BinaryTree {
    struct Node {
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        int value;
        explicit Node(const int val) : value{val} {
        }
    };

    struct NodePath {
        Node* left{};
        Node* right{};
        int value{};
        explicit NodePath(Node* node)
            : left{node->left.get()}, right{node->right.get()}, value{node->value} {
        }
    };

    struct BaseNode {
        int value;
        Node* node;
    };

public:
    BinaryTree() = default;

    template <typename Parser>
    void ParsePreOrder(Parser parser) {
        std::vector<BaseNode> nodes_array;
        nodes_array.reserve(1);
        auto element = parser();
        if (!element) {
            return;
        }

        const auto look_for_position = [&](const int number) {
            int index = static_cast<int>(std::size(nodes_array) - 1);
            while (index >= 0) {
                if (nodes_array[index].value > number) {
                    return index + 1;
                }
                --index;
            }
            return 0;
        };

        tree_root_ = std::make_unique<Node>(*element);
        nodes_array.push_back({*element, tree_root_.get()});
        tree_size_ = 1;

        for (element = parser(); element; element = parser()) {
            auto value = *element;
            auto node = std::make_unique<Node>(value);
            const auto pointer = node.get();
            ++tree_size_;

            if (nodes_array.back().value > value) {
                nodes_array.back().node->left = std::move(node);
                nodes_array.push_back({value, pointer});
                continue;
            }
            const auto position = look_for_position(value);
            int array_size = static_cast<int>(std::size(nodes_array));
            if (position == array_size) {
                return;
            } else {
                nodes_array.resize(position + 1);
                nodes_array.back().node->right = std::move(node);
                nodes_array.back().value = value;
                nodes_array.back().node = pointer;
            }
        }
    }

    const auto DiveIntoLeft(std::vector<NodePath>& path, Node* const root) {
        Node* next = root;
        while (next) {
            NodePath node_path{next};
            path.push_back(node_path);
            next = node_path.left;
        }
    }

    template <typename Callback>
    void PostOrderTraverse(Callback callback) {
        if (!tree_root_) {
            return;
        }

        std::vector<NodePath> path;
        path.reserve(tree_size_ / 2);
        DiveIntoLeft(path, tree_root_.get());

        while (!std::empty(path)) {
            const auto right = path.back().right;
            if (right) {
                path.back().right = nullptr;
                DiveIntoLeft(path, right);
                continue;
            } else {
                callback(path.back().value);
                path.pop_back();
            }
        }
    }

    template <typename Callback>
    void InOrderTraverse(Callback callback) {
        if (!tree_root_) {
            return;
        }

        std::vector<NodePath> path;
        path.reserve(tree_size_ / 2);
        DiveIntoLeft(path, tree_root_.get());

        while (!std::empty(path)) {
            callback(path.back().value);
            const auto right = path.back().right;
            path.pop_back();
            if (right) {
                DiveIntoLeft(path, right);
            }
        }
    }

private:
    std::unique_ptr<Node> tree_root_;
    int tree_size_{};
};

void ReadInputAndBuildTree(BinaryTree& tree) {
    int n_values;
    std::cin >> n_values;

    tree.ParsePreOrder([n_values]() mutable -> std::optional<int> {
        int value;
        if (!n_values) {
            return std::nullopt;
        }
        std::cin >> value;
        --n_values;
        return value;
    });
}

void WriteOutputPostOrder(BinaryTree& tree) {
    tree.PostOrderTraverse([](const auto value) { std::cout << value << " "; });
    std::cout << "\n";
}

void WriteOutputInOrder(BinaryTree& tree) {
    tree.InOrderTraverse([](const auto value) { std::cout << value << " "; });
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    BinaryTree tree;
    ReadInputAndBuildTree(tree);
    WriteOutputPostOrder(tree);
    WriteOutputInOrder(tree);

    return 0;
}
