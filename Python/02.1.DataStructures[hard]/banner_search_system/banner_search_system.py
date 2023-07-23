import heapq
from collections import defaultdict


def normalize(
        text: str
        ) -> str:
    """
    Removes punctuation and digits and convert to lower case
    :param text: text to normalize
    :return: normalized query
    """
    text = text.replace('.', ' ')
    return "".join([char.lower() for char in text if char not in r"!@#$%^&*()[]{};:,./<>?\|`~-=_+0123456789"])


def get_words(
        query: str
        ) -> list[str]:
    """
    Split by words and leave only words with letters greater than 3
    :param query: query to split
    :return: filtered and split query by words
    """
    return [s for s in query.split(' ') if len(s) > 3]


def build_index(
        banners: list[str]
        ) -> dict[str, list[int]]:
    """
    Create index from words to banners ids with preserving order and without repetitions
    :param banners: list of banners for indexation
    :return: mapping from word to banners ids
    """
    banners_words = [set(get_words(normalize(b))) for b in banners]
    index = defaultdict(list)

    for i in range(len(banners)):
        for w in banners_words[i]:
            index[w].extend([i])

    return dict(index)


def get_banner_indices_by_query(
        query: str,
        index: dict[str, list[int]]
        ) -> list[int]:
    """
    Extract banners indices from index, if all words from query contains in indexed banner
    :param query: query to find banners
    :param index: index to search banners
    :return: list of indices of suitable banners
    """
    query_words = set(get_words(normalize(query)))
    index_filtered = {k: v for k, v in index.items() if k in query_words}
    # Получаем итератор по отсортированным смерженным индексам баннеров, содержащих хоть 1 слово запроса
    merged_sorted_indices_iterator = heapq.merge(*index_filtered.values())
    count = 0
    previous_index = -1
    res = []
    for i in merged_sorted_indices_iterator:
        if previous_index == i:
            count += 1
            if count == len(query_words):
                res.append(i)
                count = 0
        else:
            count = 1
            if count == len(query_words):
                res.append(i)
                count = 0
        previous_index = i
    return res


#########################
# Don't change this code
#########################

def get_banners(
        query: str,
        index: dict[str, list[int]],
        banners: list[str]
        ) -> list[str]:
    """
    Extract banners matched to queries
    :param query: query to match
    :param index: word-banner_ids index
    :param banners: list of banners
    :return: list of matched banners
    """
    indices = get_banner_indices_by_query(query, index)
    return [banners[i] for i in indices]

#########################
