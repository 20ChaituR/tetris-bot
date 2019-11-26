import pyglet
from pyglet.window import key
import torch

import game
import train


# ========================================================================
#
#                           Initialization
#
# ========================================================================

# which model to run
# the full list of models are contained in the models folder
# training the network will generate models
model = 'models/tetris_policy_10000.pth'

# Loading the model and resetting the game state
policy = train.Policy()
policy.load_state_dict(torch.load(model))

state = game.reset()
score = 0
high_score = 0
last_move = 0

# Window constants
GAME_WIDTH = 100
UI_WIDTH = 150
GAME_HEIGHT = 180
SQUARE_WIDTH = 10
SQUARE_HEIGHT = 10

SCREEN_TITLE = "Tetris"
FONT_SIZE = 11
KEY_SIZE = 15

FPS = 1 / 2.0

# Starting up the window
window = pyglet.window.Window(GAME_WIDTH + UI_WIDTH, GAME_HEIGHT)
pyglet.gl.glClearColor(1, 1, 1, 1)
window.clear()

bot_mode = True
down_press = False


# ========================================================================
#
#                       Updating the Game State
#
# ========================================================================

# If in bot mode, get the next move from the policy network
# If not, step forward in the game
def update_frame(x):
    global state, score, high_score, last_move, bot_mode, down_press

    if bot_mode:
        action = train.select_action(state).item()
        state, reward, done = game.step(action)

        last_move = action
    else:
        state, reward, done = game.step(0)
        if down_press:
            game.active_piece, game.grid, _ = game.move_down(game.active_piece, game.grid)
            last_move = 4

    score += reward

    if done:
        game.reset()
        high_score = max(high_score, score)
        score = 0


# Draw the game and the UI elements
button_x = 0
button_y = 0
button_dx = 0
button_dy = 0
button_text = 'Stop Bot'


@window.event
def on_draw():
    global score, high_score, last_move, button_x, button_y, button_dx, button_dy, button_text, FONT_SIZE, KEY_SIZE

    window.clear()

    color_map = {
        '': (0, 0, 0),
        "L": (15, 61, 243),
        "S": (92, 195, 243),
        "I": (243, 112, 32),
        "T": (244, 240, 52),
        "B": (244, 40, 27),
        "J": (195, 68, 243),
        "Z": (39, 242, 44)}

    # Draw game
    for i in range(18):
        for j in range(10):
            x = SQUARE_WIDTH * (j + 1)
            y = GAME_HEIGHT - SQUARE_HEIGHT * i
            dx = SQUARE_WIDTH
            dy = SQUARE_HEIGHT
            pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x, y, x - dx, y, x - dx, y - dy, x, y - dy]),
                                 ('c3B', color_map[game.grid[i][j]] * 4))

    # Draw high score
    label = pyglet.text.Label('High Score: ' + str(high_score),
                              font_name='Times New Roman',
                              font_size=FONT_SIZE,
                              color=(0, 0, 0, 255),
                              x=GAME_WIDTH + UI_WIDTH // 2, y=GAME_HEIGHT - 2 * FONT_SIZE,
                              anchor_x='center', anchor_y='center')
    label.draw()

    # Draw current score
    label = pyglet.text.Label('Score: ' + str(score),
                              font_name='Times New Roman',
                              font_size=FONT_SIZE,
                              color=(0, 0, 0, 255),
                              x=GAME_WIDTH + UI_WIDTH // 2, y=GAME_HEIGHT - 4 * FONT_SIZE,
                              anchor_x='center', anchor_y='center')
    label.draw()

    # Draw bot toggle button
    button_x = GAME_WIDTH + 3 * UI_WIDTH // 4
    button_y = GAME_HEIGHT - 6 * FONT_SIZE
    button_dx = UI_WIDTH // 2
    button_dy = 2 * FONT_SIZE
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [button_x, button_y, button_x - button_dx, button_y,
                                                         button_x - button_dx, button_y - button_dy, button_x,
                                                         button_y - button_dy]), ('c3B', (200, 200, 200) * 4))

    label = pyglet.text.Label(button_text,
                              font_name='Times New Roman',
                              font_size=FONT_SIZE,
                              color=(0, 0, 0, 255),
                              x=GAME_WIDTH + UI_WIDTH // 2, y=GAME_HEIGHT - 7 * FONT_SIZE,
                              anchor_x='center', anchor_y='center')
    label.draw()

    # Draw active key
    if last_move == 1:
        x = GAME_WIDTH + (UI_WIDTH - KEY_SIZE) / 2
        y = GAME_HEIGHT - 10 * FONT_SIZE - KEY_SIZE
        dx = KEY_SIZE
        dy = KEY_SIZE
    elif last_move == 2:
        x = GAME_WIDTH + (UI_WIDTH + 3 * KEY_SIZE) / 2
        y = GAME_HEIGHT - 10 * FONT_SIZE - KEY_SIZE
        dx = KEY_SIZE
        dy = KEY_SIZE
    elif last_move == 3:
        x = GAME_WIDTH + (UI_WIDTH + KEY_SIZE) / 2
        y = GAME_HEIGHT - 10 * FONT_SIZE
        dx = KEY_SIZE
        dy = KEY_SIZE
    elif last_move == 4:
        x = GAME_WIDTH + (UI_WIDTH + KEY_SIZE) / 2
        y = GAME_HEIGHT - 10 * FONT_SIZE - KEY_SIZE
        dx = KEY_SIZE
        dy = KEY_SIZE
    if last_move == 1 or last_move == 2 or last_move == 3 or last_move == 4:
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x, y, x - dx, y, x - dx, y - dy, x, y - dy]),
                             ('c3B', (200, 200, 0) * 4))
        last_move = 0

    # Draw arrow keys
    border = 2
    x = GAME_WIDTH + (UI_WIDTH - KEY_SIZE) / 2 - border
    y = GAME_HEIGHT - 10 * FONT_SIZE - KEY_SIZE - border
    dx = KEY_SIZE - 2 * border
    dy = KEY_SIZE - 2 * border
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x, y, x - dx, y, x - dx, y - dy, x, y - dy]),
                         ('c3B', (0, 0, 0) * 4))
    x = GAME_WIDTH + (UI_WIDTH + KEY_SIZE) / 2 - border
    y = GAME_HEIGHT - 10 * FONT_SIZE - KEY_SIZE - border
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x, y, x - dx, y, x - dx, y - dy, x, y - dy]),
                         ('c3B', (0, 0, 0) * 4))
    x = GAME_WIDTH + (UI_WIDTH + 3 * KEY_SIZE) / 2 - border
    y = GAME_HEIGHT - 10 * FONT_SIZE - KEY_SIZE - border
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x, y, x - dx, y, x - dx, y - dy, x, y - dy]),
                         ('c3B', (0, 0, 0) * 4))
    x = GAME_WIDTH + (UI_WIDTH + KEY_SIZE) / 2 - border
    y = GAME_HEIGHT - 10 * FONT_SIZE - border
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x, y, x - dx, y, x - dx, y - dy, x, y - dy]),
                         ('c3B', (0, 0, 0) * 4))


# Make a move depending on the key press
@window.event
def on_key_press(symbol, modifiers):
    global last_move, down_press
    if symbol == key.LEFT:
        game.active_piece, game.grid = game.move_left(game.active_piece, game.grid)
        last_move = 1
    if symbol == key.RIGHT:
        game.active_piece, game.grid = game.move_right(game.active_piece, game.grid)
        last_move = 2
    if symbol == key.UP:
        game.active_piece, game.grid = game.rotate_piece(game.active_piece, game.grid)
        last_move = 3
    if symbol == key.DOWN:
        down_press = True


@window.event
def on_key_release(symbol, modifiers):
    global down_press
    if symbol == key.DOWN:
        down_press = False


# Toggle between bot and player mode
@window.event
def on_mouse_press(x, y, button, modifiers):
    global button_x, button_y, button_dx, button_dy, button_text, bot_mode
    if button_x - button_dx <= x and x <= button_x:
        if button_y - button_dy <= y and y <= button_y:
            if bot_mode:
                bot_mode = False
                button_text = 'Start Bot'
            else:
                bot_mode = True
                button_text = 'Stop Bot'


# Run the game at the given frames per second
pyglet.clock.schedule_interval(update_frame, FPS)
pyglet.app.run()
