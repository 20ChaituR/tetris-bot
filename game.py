import random
import numpy as np

# ========================================================================
#
#                           State Variables
#
# ========================================================================

prev_state = [[0 for _ in range(10)] for _ in range(18)]
cur_state = [[0 for _ in range(10)] for _ in range(18)]
state_size = 180  # size of grid
action_size = 4  # nothing, left, right, up

grid = [['' for _ in range(10)] for _ in range(18)]
active_piece = []
game_score = 0

# the first coordinate represents what to rotate around
pieces = [
    ["L", [(1, 4), (0, 4), (2, 4), (2, 5)]],
    ["S", [(0, 4), (0, 5), (1, 4), (1, 3)]],
    ["I", [(1, 4), (0, 4), (2, 4), (3, 4)]],
    ["T", [(1, 4), (0, 4), (1, 3), (1, 5)]],
    ["B", [(0, 4), (0, 5), (1, 4), (1, 5)]],
    ["J", [(1, 4), (0, 4), (2, 4), (2, 3)]],
    ["Z", [(0, 4), (0, 3), (1, 4), (1, 5)]]]


# ========================================================================
#
#                             Commands
#
# ========================================================================

# restart the game
def reset():
    global grid, active_piece, game_score
    grid = [['' for _ in range(10)] for _ in range(18)]
    game_score = 0
    active_piece = pieces[random.randint(0, 6)]
    for loc in active_piece[1]:
        grid[loc[0]][loc[1]] = active_piece[0]

    return get_state()


# advance one game step with the given action
def step(action):
    global grid, active_piece, game_score
    if action == 1:
        active_piece, grid = move_left(active_piece, grid)
    if action == 2:
        active_piece, grid = move_right(active_piece, grid)
    if action == 3:
        active_piece, grid = rotate_piece(active_piece, grid)

    active_piece, grid, poss = move_down(active_piece, grid)

    score = 0

    if not poss:
        grid, score = clear_lines(grid)

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

    return get_state(), score, False


# returns the state of the game, which is a reshaped form of grid
def get_state():
    global grid, active_piece, game_score, prev_state, cur_state

    cur_state = np.zeros((18, 10))
    for i in range(18):
        for j in range(10):
            cur_state[i][j] = 0 if grid[i][j] == '' else 1

    state = np.zeros((18, 10))
    for i in range(18):
        for j in range(10):
            state[i][j] = cur_state[i][j] - prev_state[i][j]

    prev_state = cur_state

    return state.reshape((1, 1, 18, 10))


# ========================================================================
#
#                             Moves
#
# ========================================================================

# rotates given piece clockwise if possible
def rotate_piece(piece, grid):
    if piece[0] == 'B':
        return piece, grid

    midLoc = piece[1][0]

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


# moves given piece to the left if possible
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


# moves given piece to the right if possible
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


# moves given piece down if possible
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


# clears all lines in the grid and returns the score
def clear_lines(grid):
    count = 0
    for r in range(len(grid)):
        isLine = True
        for c in range(len(grid[0])):
            if grid[r][c] == '':
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
        score = 1
    elif count == 1:
        score = 400
    elif count == 2:
        score = 1000
    elif count == 3:
        score = 3000
    else:
        score = 12000
    return grid, score
