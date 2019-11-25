import pyglet
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical

import game

model = 'models/tetris_policy_100.pth'


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = game.state_size
        self.action_space = game.action_size

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.gamma = 0.99

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)


def select_action(state):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    c = Categorical(state)
    action = c.sample()
    return action


policy = Policy()
policy.load_state_dict(torch.load(model))

state = game.reset()

# Set constants for the screen size
SCREEN_WIDTH = 100
SCREEN_HEIGHT = 180
SQUARE_WIDTH = 10
SQUARE_HEIGHT = 10
SCREEN_TITLE = "Tetris"

window = pyglet.window.Window(SCREEN_WIDTH, SCREEN_HEIGHT)
window.clear()


def update_frame(x, y):
    global state
    action = select_action(state)
    state, _, done = game.step(action.item())

    if done:
        game.reset()


@window.event
def on_draw():
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

    for i in range(18):
        for j in range(10):
            x = SCREEN_WIDTH - SQUARE_WIDTH * j
            y = SCREEN_HEIGHT - SQUARE_HEIGHT * i
            dx = SQUARE_WIDTH
            dy = SQUARE_HEIGHT
            pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x, y, x - dx, y, x - dx, y - dy, x, y - dy]),
                                 ('c3B', color_map[game.grid[i][j]] * 4))


pyglet.clock.schedule(update_frame, 1 / 10.0)
pyglet.app.run()
