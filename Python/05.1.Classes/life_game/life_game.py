from typing import List
from collections import Counter, defaultdict


class LifeGame(object):
    """
    Class for Game life
    """
    def __init__(self, ocean: List[List[int]]):
        self.ocean = ocean
        self.rows = len(ocean)
        self.cols = len(ocean[0])
        self.mapping = {0: 'empty',
                        1: 'rock',
                        2: 'fish',
                        3: 'shrimp'}

    def _get_neighbours(self, i: int, j: int) -> dict[int, int]:
        neighbours = []
        for ii in [i-1, i, i+1]:
            for jj in [j-1, j, j+1]:
                if (
                        not (ii == i and jj == j)
                        and (0 <= ii < self.rows)
                        and (0 <= jj < self.cols)
                ):
                    neighbours.append(self.ocean[ii][jj])
        return Counter(neighbours)

    def _too_tight_or_alone(self, kind: int, i: int, j: int) -> dict[tuple[int, int], int]:
        """
        For fish and shrimp
        """
        neighbours: dict[int, int] = defaultdict(int)
        neighbours.update(self._get_neighbours(i, j))
        res = {}
        if (neighbours[kind] >= 4) or (neighbours[kind] <= 1):
            res.update({(i, j): 0})
        return res

    def _born(self, i: int, j: int) -> dict[tuple[int, int], int]:
        """
        For fish and shrimp
        """
        neighbours: dict[int, int] = defaultdict(int)
        neighbours.update(self._get_neighbours(i, j))
        res = {}
        if neighbours[2] == 3:
            res.update({(i, j): 2})
        elif neighbours[3] == 3:
            res.update({(i, j): 3})
        return res

    def get_next_generation(self) -> List[List[int]]:
        all_res = {}
        for i in range(self.rows):
            for j in range(self.cols):
                if (
                        (0 <= i < self.rows)
                        and (0 <= j < self.cols)
                ):
                    kind = self.ocean[i][j]
                    if kind in [2, 3]:
                        res = self._too_tight_or_alone(kind, i, j)
                        all_res.update(res)
                    if kind == 0:
                        res = self._born(i, j)
                        all_res.update(res)
        for k, v in all_res.items():
            self.ocean[k[0]][k[1]] = v
        return self.ocean
