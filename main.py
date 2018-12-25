from game.connect_four import Game
from NN.model import Model
import game_runner

game = Game()

#new_game.start()

model = Model()

while(True):
    message = "Select action:\n1. Train a new AI\n2. Keep training the existing AI\n3. Play the AI\n4. Exit\n"
    action = input(message)
    message = "Type the name of AI\n"
    if action == '1':
        name = input(message)
        model.train(name)
    elif action == '2':
        name = input(message)
        model.load(name)
        model.train(name)
    elif action == '3':
        name = input(message)
        game_runner.play_AI(model, name)
    else:
        break

print("Done running the program")
