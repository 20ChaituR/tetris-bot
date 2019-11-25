import random
from time import sleep

import ai
import game

moves = [0, 0, 0, 3, 0, 0, 1, 1]

state = game.reset()
for i in range(2000):
    game.render()
    f = True
    action = moves[i]
    state, reward, done = game.step(action)
    if done:
        break

    sleep(0.1)
