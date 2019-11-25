import math
import random

import numpy as np

state_size = 180
action_size = 4  # nothing, left, right, up
grid = [['' for j in range(10)] for i in range(18)]
active_piece = []
game_score = 0

pieces = [
    ["L", [(0, 4), (1, 4), (2, 4), (2, 5)]],
    ["S", [(1, 5), (1, 4), (2, 4), (2, 3)]],
    ["I", [(0, 4), (1, 4), (2, 4), (3, 4)]],
    ["T", [(1, 4), (2, 4), (2, 3), (2, 5)]],
    ["B", [(0, 4), (0, 5), (1, 4), (1, 5)]],
    ["J", [(0, 4), (1, 4), (2, 4), (2, 3)]],
    ["Z", [(1, 3), (1, 4), (2, 4), (2, 5)]]]


def reset():
    global grid, active_piece, game_score
    grid = [['' for _ in range(10)] for _ in range(18)]
    game_score = 0
    active_piece = pieces[random.randint(0, 6)]
    for loc in active_piece[1]:
        grid[loc[0]][loc[1]] = active_piece[0]

    return get_state()


def step(action):
    global grid, active_piece, game_score
    if action == 1:
        active_piece, grid = move_left(active_piece, grid)
    if action == 2:
        active_piece, grid = move_right(active_piece, grid)
    if action == 3:
        active_piece, grid = rotate_piece(active_piece, grid)

    active_piece, grid, poss = move_down(active_piece, grid)

    score = 1

    if not poss:
        active_piece = pieces[random.randint(0, 6)]
        lose = False
        for loc in active_piece[1]:
            if grid[loc[0]][loc[1]]:
                lose = True
                break
            else:
                grid[loc[0]][loc[1]] = active_piece[0]

        if lose:
            return get_state(), 0, True

        grid, score = clear_lines(grid)

    return get_state(), score, False


def get_state():
    global grid, active_piece, game_score
    piece_map = {
        '': 0,
        'L': 1,
        'S': 2,
        'I': 3,
        'T': 4,
        'B': 5,
        'J': 6,
        'Z': 7}

    st = np.zeros((18, 10))
    for i in range(18):
        for j in range(10):
            st[i][j] = piece_map[grid[i][j]]

    return st.reshape(180)


def rotate_piece(piece, grid):
    midLoc = piece[1][0]
    if piece[0] == 'B':
        return piece, grid
    avgX = 0
    avgY = 0
    for loc in piece[1]:
        avgX += loc[0]
        avgY += loc[1]
    avgX /= 4
    avgY /= 4
    if piece[0] == 'L' or piece[0] == 'J' or piece[0] == 'T':
        midLoc = (round(avgX), round(avgY))
    if piece[0] == 'I':
        midLoc = (int(avgX), int(avgY))
    if piece[0] == 'Z' or piece[0] == 'S':
        midLoc = (math.ceil(avgX), math.ceil(avgY))

    newPieceLoc = [loc for loc in piece[1]]
    for i in range(len(piece[1])):
        delX = piece[1][i][0] - midLoc[0]
        delY = piece[1][i][1] - midLoc[1]
        newPieceLoc[i] = (midLoc[0] + delY, midLoc[1] - delX)

    for loc in piece[1]:
        grid[loc[0]][loc[1]] = ''

    correctMovement = True

    for loc in newPieceLoc:
        if loc[0] < 0 or loc[1] < 0 or loc[0] >= len(grid) or loc[1] >= len(grid[0]):
            correctMovement = False
            break
        if grid[loc[0]][loc[1]]:
            correctMovement = False
            break

    if not correctMovement:
        for loc in piece[1]:
            grid[loc[0]][loc[1]] = piece[0]
        return piece, grid

    for loc in newPieceLoc:
        grid[loc[0]][loc[1]] = piece[0]
    return [piece[0], newPieceLoc], grid


def move_left(piece, grid):
    newPieceLoc = [loc for loc in piece[1]]
    for i in range(len(piece[1])):
        newPieceLoc[i] = (piece[1][i][0], piece[1][i][1] - 1)

    for loc in piece[1]:
        grid[loc[0]][loc[1]] = ''

    correctMovement = True

    for loc in newPieceLoc:
        if loc[0] < 0 or loc[1] < 0 or loc[0] >= len(grid) or loc[1] >= len(grid[0]):
            correctMovement = False
            break
        if grid[loc[0]][loc[1]]:
            correctMovement = False
            break

    if not correctMovement:
        for loc in piece[1]:
            grid[loc[0]][loc[1]] = piece[0]
        return piece, grid

    for loc in newPieceLoc:
        grid[loc[0]][loc[1]] = piece[0]
    return [piece[0], newPieceLoc], grid


def move_right(piece, grid):
    newPieceLoc = [loc for loc in piece[1]]
    for i in range(len(piece[1])):
        newPieceLoc[i] = (piece[1][i][0], piece[1][i][1] + 1)

    for loc in piece[1]:
        grid[loc[0]][loc[1]] = ''

    correctMovement = True

    for loc in newPieceLoc:
        if loc[0] < 0 or loc[1] < 0 or loc[0] >= len(grid) or loc[1] >= len(grid[0]):
            correctMovement = False
            break
        if grid[loc[0]][loc[1]]:
            correctMovement = False
            break

    if not correctMovement:
        for loc in piece[1]:
            grid[loc[0]][loc[1]] = piece[0]
        return piece, grid

    for loc in newPieceLoc:
        grid[loc[0]][loc[1]] = piece[0]
    return [piece[0], newPieceLoc], grid


def move_down(piece, grid):
    newPieceLoc = [loc for loc in piece[1]]
    for i in range(len(piece[1])):
        newPieceLoc[i] = (piece[1][i][0] + 1, piece[1][i][1])

    for loc in piece[1]:
        grid[loc[0]][loc[1]] = ''

    correctMovement = True

    for loc in newPieceLoc:
        if loc[0] < 0 or loc[1] < 0 or loc[0] >= len(grid) or loc[1] >= len(grid[0]):
            correctMovement = False
            break
        if grid[loc[0]][loc[1]]:
            correctMovement = False
            break

    if not correctMovement:
        for loc in piece[1]:
            grid[loc[0]][loc[1]] = piece[0]
        return piece, grid, False

    for loc in newPieceLoc:
        grid[loc[0]][loc[1]] = piece[0]
    return [piece[0], newPieceLoc], grid, True


def clear_lines(grid):
    count = 0
    for r in range(len(grid)):
        isLine = True
        for c in range(len(grid[0])):
            if not grid[r][c]:
                isLine = False
                break
        if isLine:
            count += 1
            for c in range(len(grid[0])):
                grid[r][c] = ''
            for nr in range(r, 0, -1):
                for c in range(len(grid[0])):
                    grid[nr][c] = grid[nr - 1][c]
    if count == 0:
        score = 0
    elif count == 1:
        score = 40
    elif count == 2:
        score = 100
    elif count == 3:
        score = 300
    else:
        score = 1200
    return grid, score
