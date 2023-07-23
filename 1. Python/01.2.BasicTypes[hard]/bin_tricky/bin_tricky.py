import typing as tp


def find_median(nums1: tp.Sequence[int], nums2: tp.Sequence[int]) -> float:
    """
    Find median of two sorted sequences. At least one of sequences should be not empty.
    :param nums1: sorted sequence of integers
    :param nums2: sorted sequence of integers
    :return: middle value if sum of sequences' lengths is odd
             average of two middle values if sum of sequences' lengths is even
    """
    a, b = len(nums1), len(nums2)
    if a > b:
        return find_median(nums2, nums1)
    left, middle, right = 0, (a + b + 1) // 2, a

    while left <= right:
        i = (left + right) // 2
        j = middle - i
        left_max_a = -999 if (i == 0) else nums1[i - 1]
        right_min_x = 999 if (i == a) else nums1[i]
        left_max_b = -999 if (j == 0) else nums2[j - 1]
        right_min_b = 999 if (j == b) else nums2[j]

        if (left_max_a <= right_min_b) and (left_max_b <= right_min_x):
            if (a + b) % 2 == 0:
                return (max(left_max_a, left_max_b) +
                        min(right_min_x, right_min_b)) / 2.0
            else:
                return float(max(left_max_a,
                                 left_max_b))
        elif left_max_a > right_min_b:
            right = i - 1
        else:
            left = i + 1

    assert False, "Not reachable"
