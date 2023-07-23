// Copyright [2021] <Fedor Erin>
#include <iostream>
#include <string>
#include <stack>

bool is_correct_pair(char left, char right) {
    return (left == '(' && right == ')') ||
           (left == '[' && right == ']') ||
           (left == '{' && right == '}');
}

int correct_length(std::string sequence) {
    bool correct_flag = true;
    int correct_len = 0;
    std::stack<char> stack;

    for (char& bracket : sequence) {
        if (bracket == '(' || bracket == '[' || bracket == '{') {
            stack.push(bracket);
            correct_len++;
        } else if (bracket == ')' || bracket == ']' || bracket == '}') {
            if (!stack.empty() && is_correct_pair(stack.top(), bracket)) {
                stack.pop();
                correct_len++;
            } else {
                correct_flag = false;
                break;
            }
        }
    }
    if (correct_flag && stack.empty()) {
        return -1;
    } else {
        return correct_len;
    }
}

int main() {
    std::string input;
    getline(std::cin, input);

    int result = correct_length(input);

    if (result == -1) {
        std::cout << "CORRECT" << "\n";
    } else {
        std::cout << result << "\n";
    }
    return 0;
}
