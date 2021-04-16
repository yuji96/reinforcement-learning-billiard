from enum import IntEnum

from billiards.obstacles import InfiniteWall


WIDTH = 112
LENGTH = 224
CUSHIONS = [
    InfiniteWall((0, 0), (LENGTH, 0)),  # bottom side
    InfiniteWall((LENGTH, 0), (LENGTH, WIDTH)),  # right side
    InfiniteWall((LENGTH, WIDTH), (0, WIDTH)),  # top side
    InfiniteWall((0, WIDTH), (0, 0)),  # left side
]
RADIUS = 2.85

INITIAL_WHITE_POS = (LENGTH * 0.25, WIDTH * 3/8)


class Ball(IntEnum):
    NONE = -1
    YELLOW = 0
    RED = 1
    WHITE = 2
