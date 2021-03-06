import tensorflow as tf
from NN import parameter as pm
import numpy as np
from game.connect_four import Game
import random

class Model:

    def __init__(self):
        self.input_state = tf.placeholder(tf.float32, [None, 6, 7, 1]) #Where data comes in
        #the shape is [Batch_size, h, w, channel]
        #TODO: shpae might have to be opposite, as in it could be 7, 6

        #Conv layer
        conv1 = tf.layers.conv2d(
            inputs = self.input_state,
            filters = pm.filter_num,
            kernel_size = [4, 4],
            padding = 'same',
            use_bias = True
            #activation = tf.nn.relu
            )

        conv1_out = tf.layers.Flatten()(conv1)
        #Dense layers
        dense1 = tf.layers.dense(
            inputs = conv1_out,
            units = 128,
#            activation = 'relu',
            use_bias = True
            )
        dropout1 = tf.layers.dropout(
            inputs = dense1,
            rate = pm.dr
            )
        dense2 = tf.layers.dense(
            inputs = dropout1,
            units = 128,
#            activation = 'relu',
            use_bias = True
            )
        dropout2 = tf.layers.dropout(
            inputs = dense2,
            rate = pm.dr
            )
        dense3 = tf.layers.dense(
            inputs = dropout2,
            units = 64,
        #     activation = 'relu',
            use_bias = True
            )
        dense4 = tf.layers.dense(
            inputs = dense3,
            units = 32,
        #     activation = 'relu',
            use_bias = True
            )

        self.out = tf.layers.dense(
            inputs = dense4,
            units = 7,
#            activation = 'relu',
            use_bias = True
            )

        #Tensor to save the model
        self.saver = tf.train.Saver()

        #Tensors to calculate the loss and train
        self.output_ph = tf.placeholder(tf.float32, [pm.batch_size, 7])
        loss = tf.losses.mean_squared_error(
            labels = self.output_ph,
            predictions = self.out
            )
        self.train_step = tf.train.GradientDescentOptimizer(pm.lr).minimize(loss)


        self.init_OP = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(self.init_OP)

    def train(self, name):
        games_count = 0

        #sess = tf.Session()
        #sess.run(self.init_OP)
        while games_count < pm.game_number: #keep training until it reaches epoch limit
            #Play batch number of games first
            D = list() #list to keep record:state, action, new state after action, reward
            while len(D) < pm.batch_size:
                games_count += 1
                env = Game()
                game_status = 0 #Game's just started
                while(game_status == 0):
                    #player1
                    state, game_status = env.get_state(1)
                    if np.random.rand() < pm.el:
                        action = np.random.randint(0, 7, size=1)[0] #TODO: make sure 0 to 6 is randomly selected
                    else:
                        #get_state returns the board and game status (0 = undecided, 1 = player 1 won...)
                        action = self.play(state)

                    env.input(action, 1) #Execute the action as player 1
                    print("Player 1 - Action: " + str(action))
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
                        else:
                            action = self.play(state)

                        env.input(action, 2) #Execute the action as player 2
                        print("Player 2 - Action: " + str(action))
                        env.display()
                        new_state, game_status = env.get_state(2) #Observe the env
                        new_data = [state, action, new_state]

                        if game_status == 2:
                            new_data.append(1)
                        elif game_status == 1:
                            new_data.append(-1)
                        else:
                            new_data.append(0)
                        D.append(new_data)
                #---End of a game---
                print("This is the end result")
                env.display()

            print("Done playing")
            print("Game count: " + str(games_count))
            #---End of a batch---

            #Training with the data from playing
            print("Started training with a batch")
            batch = random.sample(D, pm.batch_size)
            input = np.zeros([pm.batch_size, 6, 7, 1])
            output = np.zeros([pm.batch_size, 7])
            loss = list()

            for i in range(pm.batch_size):
                state = self.reshape(batch[i][0]) #State on which the AI made decision on
                action = batch[i][1] #Decision AI made
                new_state = self.reshape(batch[i][2]) #State as the cosequence of action
                reward = batch[i][3] #Reward AI got after the action

                if state == new_state:
                    reward = -50

                #state[0] because state is a batch -> have to reduce one dimension
                input[i] = state[0]

                output[i] = self.sess.run(
                    self.out,
                    feed_dict = {self.input_state: state})[0]
                reward += pm.q_gamma * np.argmax(self.sess.run(
                    self.out,
                    feed_dict = {self.input_state: new_state})[0])
                output[i][action] = reward

            self.sess.run(self.train_step, feed_dict = {self.input_state: input, self.output_ph: output})

            D.clear() #clear it for next batch
            print("Done training with a batch")
            print(str(games_count/pm.game_number * 100) + "% of training done")

            if games_count % 50 == 0:
                self.saver.save(self.sess, "models/" + name + "-checkpoint.ckpt")
                print("Checkpoint model saved")
        #Save the trained model
        self.saver.save(self.sess, "models/" + name + ".ckpt")
        print("Done")



    def load(self, name): #This load already trained NN
        #with tf.Session() as sess:
        self.saver.restore(self.sess, "models/" + name + ".ckpt")
        print("Model Restored from Checkpoint")

    def reshape(self, s):
        s_feed = list()
        temp = list()
        for i in range(6):
            temp_row = list()
            for j in range(7):
                temp_row.append([s[i][j]])
            temp.append(temp_row)
        s_feed.append(temp)

        return s_feed

    def play(self, s): #input state, return action
        #with tf.Session() as sess:
            #sess.run(self.init_OP)
        s_feed = self.reshape(s)
        Q = self.sess.run(self.out, feed_dict ={self.input_state: s_feed}) #TODO: Call the output of NN
        print(Q[0])
        print("AI Made a decision: " + str(np.argmax(Q[0])))
        action = self.elim_actions(s, Q[0])
        return np.argmax(action)

    def close(self):
        self.sess.close()
