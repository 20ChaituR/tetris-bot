import pyglet
import torch

import game
import train

model = 'models/tetris_policy_500.pth'


policy = train.Policy()
policy.load_state_dict(torch.load(model))

state = game.reset()
score = 0
high_score = 0

last_move = 0

# Set constants for the screen size
GAME_WIDTH = 100
UI_WIDTH = 150
GAME_HEIGHT = 180
SQUARE_WIDTH = 10
SQUARE_HEIGHT = 10

SCREEN_TITLE = "Tetris"

window = pyglet.window.Window(GAME_WIDTH + UI_WIDTH, GAME_HEIGHT)
pyglet.gl.glClearColor(1,1,1,1)
window.clear()


def update_frame(x, y):
    global state, score, high_score, last_move

    action = train.select_action(state)
    state, reward, done = game.step(action.item())
    score += reward
    last_move = action.item()

    if done:
        game.reset()
        high_score = max(high_score, score)
        score = 0


@window.event
def on_draw():
    global score, high_score, last_move

    window.clear()

    color_map = {
        '': (0,0,0),
        "L": (15, 61, 243),
        "S": (92, 195, 243),
        "I": (243, 112, 32),
        "T": (244, 240, 52),
        "B": (244, 40, 27),
        "J": (195, 68, 243),
        "Z": (39, 242, 44)}

    for i in range(18):
        for j in range(10):
            x = SQUARE_WIDTH * (j + 1)
            y = GAME_HEIGHT - SQUARE_HEIGHT * i
            dx = SQUARE_WIDTH
            dy = SQUARE_HEIGHT
            pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x, y, x - dx, y, x - dx, y - dy, x, y - dy]),
                                 ('c3B', color_map[game.grid[i][j]] * 4))

    font_size = 11

    label = pyglet.text.Label('High Score: ' + str(high_score),
                              font_name='Times New Roman',
                              font_size=font_size,
                              color = (0,0,0,255),
                              x=GAME_WIDTH + UI_WIDTH // 2, y=GAME_HEIGHT - 2 * font_size,
                              anchor_x='center', anchor_y='center')
    label.draw()

    label = pyglet.text.Label('Score: ' + str(score),
                              font_name='Times New Roman',
                              font_size=font_size,
                              color=(0, 0, 0, 255),
                              x=GAME_WIDTH + UI_WIDTH // 2, y=GAME_HEIGHT - 4 * font_size,
                              anchor_x='center', anchor_y='center')
    label.draw()

    x = GAME_WIDTH + 3 * UI_WIDTH // 4
    y = GAME_HEIGHT - 6 * font_size
    dx = UI_WIDTH // 2
    dy = 2 * font_size
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x, y, x - dx, y, x - dx, y - dy, x, y - dy]),
                         ('c3B', (200,200,200) * 4))

    label = pyglet.text.Label('Stop Bot',
                              font_name='Times New Roman',
                              font_size=font_size,
                              color=(0, 0, 0, 255),
                              x=GAME_WIDTH + UI_WIDTH // 2, y=GAME_HEIGHT - 7 * font_size,
                              anchor_x='center', anchor_y='center')
    label.draw()

    if last_move == 1:
        x = GAME_WIDTH + (UI_WIDTH + font_size) / 2
        y = GAME_HEIGHT - 10 * font_size
        dx = 11
        dy = 11
    elif last_move == 2:
        x = GAME_WIDTH + (UI_WIDTH - font_size) / 2
        y = GAME_HEIGHT - 10 * font_size
        dx = 11
        dy = 11
    elif last_move == 3:
        x = GAME_WIDTH + (UI_WIDTH + 2 * font_size) / 2
        y = GAME_HEIGHT - 9 * font_size
        dx = 11
        dy = 11
    if last_move == 1 or last_move == 2 or last_move == 3:
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x, y, x - dx, y, x - dx, y - dy, x, y - dy]),
                             ('c3B', (255,255,0) * 4))


pyglet.clock.schedule(update_frame, 1 / 10.0)
pyglet.app.run()
