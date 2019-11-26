import pyglet
from pyglet.window import Window
from pyglet.window import key

window = Window()
keyboard = key.KeyStateHandler()
window.push_handlers(keyboard)

@window.event
def on_key_press(symbol, modifiers):
    print(keyboard[key.DOWN])

pyglet.app.run()