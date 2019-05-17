import keras
from game.connect_four import Game
from NN import parameter as pm
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, Flatten
import numpy as np
import random


def data_mirror(board):
    #This returns the board mirrored horizontally in order to increase data
    #TODO: implement this.
    i = 0

def reshape(s):
    s_feed = list()
    temp = list()
    for i in range(6):
        temp_row = list()
        for j in range(7):
            temp_row.append([s[i][j]])
        temp.append(temp_row)
    s_feed.append(temp)

    return s_feed


class Model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(6, 7, 1)))
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(1024, activation = 'relu'))
        self.model.add(Dense(7, activation = 'relu'))
        self.model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    def play(self, board):
        #This func predicts the best decision based on board input
        board = reshape(board)
        board = np.array(board)
        q_pred = self.model.predict(board)
        return np.argmax(q_pred)

    def load(self, name):
        #TODO: load the model with the name
        self.model = load_model(name + '.h5')

    def save(self, name):
        self.model.save(name + '.h5')

    def train(self, name):
        #This func trains the model

        self.model.save('copy_temp.h5')
        oppo = Model()
        oppo.load('copy_temp')
        games_count = 0
        training_turn = 1

        while games_count < pm.game_number: #keep training until it reaches epoch limit
            #Play batch number of games first
            D = list() #list to keep record:state, action, new state after action, reward
            while len(D) < pm.batch_size: #Keep playing the game until u have more than batch_size data
                games_count += 1
                env = Game()
                game_status = 0 #Game's just started

                #This if struct randomly decides which model plays first
                if np.random.randint(0, 2) == 0:
                    fir_player = self
                    sec_player = oppo
                else:
                    fir_player = oppo
                    sec_player = self

                while(game_status == 0):
                    #player1
                    state, game_status = env.get_state(1)
                    if np.random.rand() < pm.el:
                        action = np.random.randint(0, 7, size=1)[0] #TODO: make sure 0 to 6 is randomly selected
                        #print("Player 1 - Action: " + str(action) + " (random)")
                    else:
                        #get_state returns the board and game status (0 = undecided, 1 = player 1 won...)
                        action = fir_player.play(state)
                        #print("Player 1 - Action: " + str(action))


                    env.input(action, 1) #Execute the action as player 1
                    if games_count % 100 == 0:
                        env.display()
                    new_state, game_status = env.get_state(1) #Observe the env
                    new_data = [state, action, new_state]
                    #Append the reward to data list
                    if game_status == 1:
                        new_data.append(1)
                    elif game_status == 2:
                        new_data.append(-1)
                    else:
                        new_data.append(0)
                    D.append(new_data)
                    #End of player1 --------------------------------------

                    #Player 2
                    state, game_status = env.get_state(2)
                    if game_status == 0:
                        if np.random.rand() < pm.el:
                            action = np.random.randint(0, 7, size=1)[0]
                            #print("Player 2 - Action: " + str(action) + " (random)")
                        else:
                            action = sec_player.play(state)
                            #print("Player 2 - Action: " + str(action))
                        if games_count % 100 == 0:
                            env.display()
                        env.input(action, 2) #Execute the action as player 2
                        #if games_count % 100 == 0:
                            #env.display()
                        new_state, game_status = env.get_state(2) #Observe the env
                        new_data = [state, action, new_state]

                        if game_status == 2:
                            new_data.append(1)
                        elif game_status == 1:
                            new_data.append(-1)
                        else:
                            new_data.append(0)
                        D.append(new_data)
                    #End of player 2 Turn ----------------------
                #---End of a game---
                if games_count % 100 == 0:
                    print("This is the end result")
                    env.display()

            print("Done playing")
            print("Game count: " + str(games_count))
            #---End of collecting a batch---

            #Training with the data from playing
            print("Started training with a batch")
            batch = random.sample(D, pm.batch_size)

            if training_turn == 1:
                train_model = self
                training_turn = 2
            else:
                train_model = oppo
                training_turn = 1

            input = np.zeros([pm.batch_size, 6, 7, 1])
            output = np.zeros([pm.batch_size, 7])

            for i in range(pm.batch_size):
                state = batch[i][0] #State on which the AI made decision on
                state_re = reshape(state)
                action = batch[i][1] #Decision AI made
                new_state = batch[i][2] #State as the cosequence of action
                reward = batch[i][3] #Reward AI got after the action

                #state[0] because state is a batch -> have to reduce one dimension
                input[i] = state_re[0]

                output[i] = train_model.play(state)
                reward += pm.q_gamma * np.amax(train_model.play(new_state))
                output[i][action] = reward
            train_model.model.fit(input, output)
            print("Done training with a batch")


            D.clear() #clear it for next batch
            print("Done training with a batch")
            print(str(games_count/pm.game_number * 100) + "% of training done")

            if games_count % 50 == 0:
                self.save(name + '-checkpoint')
                print("Checkpoint model saved")
        #Save the trained model
        self.save(name)
        print("Done")
