import typing as tp


def find_value(nums: tp.Union[list[int], range], value: int) -> bool:
    """
    Find value in sorted sequence
    :param nums: sequence of integers. Could be empty
    :param value: integer to find
    :return: True if value exists, False otherwise
    """
    left = 0
    right = len(nums) - 1
    if nums:
        while right - left > 0:
            middle = (left + right) // 2
            if nums[middle] >= value:
                right = middle
            else:
                left = middle + 1
        return nums[left] == value
    else:
        return False
