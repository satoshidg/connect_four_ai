from game.connect_four import Game
#from NN.model import Model
from NN.model import Model
from NN.network import Network
import game_runner
import numpy as np

game = Game()

model = Model()
#new_game.start()

network = Network()

while(True):
    message = "Select action:\n1. Train a new AI\n2. Keep training the existing AI\n3. Play the AI\n4. Watch 2 AI play\n5. Exit\n"
    action = input(message)
    message = "Type the name of AI\n"
    if action == '0':
        game_runner.play_2()
    elif action == '1': #Training a new AI
        name = input(message)
        #network.train(name, False) #False as in not retraining
        print(model.model)
        model.train(name)
    elif action == '2': #Retraining an existing AI
        name = input(message)
        model.load(name)
        model.train(name)
    elif action == '3': #Play an AI
        name = input(message)
        model.load(name)
        game_runner.play_AI(model, name)
    elif action == '4': #Watch AI play each other
        print("Not yet implemented")
        name = input(message)
        name = input(message)
        print("Not yet i")
        #TODO: implement here
    elif action == '5':
        break
    else:
        print("Invalid Input")
    print("\n")

network.close()
print("Quitting the program")
