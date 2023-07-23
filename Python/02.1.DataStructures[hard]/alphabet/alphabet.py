import enum


class Status(enum.Enum):
    NEW = 0
    EXTRACTED = 1
    FINISHED = 2


def extract_alphabet(
        graph: dict[str, set[str]]
        ) -> list[str]:
    """
    Extract alphabet from graph
    :param graph: graph with partial order
    :return: alphabet
    """
    # Tarjan algo
    colors = {k: Status.NEW for k in graph.keys()}
    result_reversed: list[str] = []

    def step(node: str) -> None:
        if colors[node] == Status.FINISHED:
            pass
        elif colors[node] == Status.EXTRACTED:
            assert False, "Loop found!"
        else:
            colors[node] = Status.EXTRACTED
            for leaf in graph[node]:
                step(leaf)
            colors[node] = Status.FINISHED
            result_reversed.extend([node])

    for k in graph.keys():
        step(k)

    return result_reversed[::-1]


def build_graph(
        words: list[str]
        ) -> dict[str, set[str]]:
    """
    Build graph from ordered words. Graph should contain all letters from words
    :param words: ordered words
    :return: graph
    """
    def padding(str1: str, str2: str) -> tuple[str, str]:
        if len(str1) >= len(str2):
            str2 = str2.ljust(len(str1))
        else:
            str1 = str1.ljust(len(str2))
        return str1, str2

    def default_add(tree: dict[str, set[str]], node: str) -> None:
        if node not in tree and node != ' ':
            tree[node] = set()

    graph: dict[str, set[str]] = dict()
    if len(words) == 1:
        graph = {letter: set() for letter in words[0]}
    if len(words) > 1:
        for i in range(len(words)-1):
            w1, w2 = padding(words[i], words[i+1])
            prefix1, prefix2 = '', ''
            for l1, l2 in zip(list(w1), list(w2)):
                if l1 != ' ' and l2 != ' ':
                    if l1 != l2 and prefix1 == prefix2:
                        default_add(graph, l2)
                        if l1 not in graph:
                            graph[l1] = {l2}
                        else:
                            graph[l1] = graph[l1].union({l2})
                    else:
                        default_add(graph, l1)
                        default_add(graph, l2)
                default_add(graph, l1)
                default_add(graph, l2)
                prefix1 += l1
                prefix2 += l2
    return graph

#########################
# Don't change this code
#########################


def get_alphabet(
        words: list[str]
        ) -> list[str]:
    """
    Extract alphabet from sorted words
    :param words: sorted words
    :return: alphabet
    """
    graph = build_graph(words)
    return extract_alphabet(graph)

#########################
