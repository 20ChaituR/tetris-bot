# Tetris Bot

A Reinforcement-Learning Policy network that can play Tetris. This bot uses pytorch and is inspired by [this article](https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf) implementing a policy gradient for the Cart-Pole problem.

## Training

The learning rate, gamma factor, and number of episodes can be configured when training. To train, just run `train.py`. The finished model will be saved in the models folder.

## Simulating a Game

Run `run.py` to simulate a game. The model's filename should be given at the start. While playing a game, you can toggle the bot on and off and take control with the arrow keys.