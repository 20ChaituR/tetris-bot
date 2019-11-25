from graphics import *

called = False
win = None

SCREEN_WIDTH = 200
SCREEN_HEIGHT = 360
SQUARE_WIDTH = 10
SQUARE_HEIGHT = 10


def initialize():
    global called, win
    win = GraphWin('Tetris', SCREEN_WIDTH, SCREEN_HEIGHT)


def draw_grid(grid):
    global called
    if not called:
        initialize()
        called = True

    color_map = {
        '': "black",
        "L": "blue",
        "S": "cyan",
        "I": "orange",
        "T": "yellow",
        "B": "red",
        "J": "purple",
        "Z": "green"}

    for i in range(18):
        for j in range(10):
            square = Rectangle(Point(SQUARE_WIDTH * j, SQUARE_HEIGHT * i),
                               Point(SQUARE_WIDTH * (j + 1), SQUARE_HEIGHT * (i + 1)))
            square.setFill(color_map[grid[i][j]])
            square.draw(win)
