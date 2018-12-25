#from NN.model_old import Model
from NN.model import Model
from game.connect_four import Game
import tensorflow as tf

def play_AI(cpu, name):
    sess = tf.Session()
    cpu.load(name)

    game = Game()

    player = 1
    print("Starting game...")
    game.display()
    while(game.result() == 0):
        message = "Player" + str(player)
        if player == 1:
            message += "(X)"
            message += "'s turn\nEnter column index: "
            column = input(message)
        else:
            print("AI's turn")
            column = cpu.play(game.board)
        game.mark(int(column), player)
        if player == 1:
            player = 2
        else:
            player = 1
        game.display()
    result = game.result()
    if result == -1:
        print("The board is full, game is Tie!")
    elif result == 1:
        print("You won!")
    else:
        print("You lost!")
    sess.close()
