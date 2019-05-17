from array import *
#from NN import model

class Game:
    # Initializing gameself.board
    # 0 = unused, 1 = player 1, 2 = player 2.
    # 6rows, 7 columns

    def __init__(self):
        self.board = []
        for i in range(6):
            self.board.append([0,0,0,0,0,0,0])

    def display(self):
      for i in range(6):
          row = "|"
          for j in range(7):
              if self.board[i][j] == 0:
                  row += " |"
              elif self.board[i][j] == 1:
                  row += "X|"
              else:
                  row += "O|"
          print(row)
    #            print("-------")

    def mark(self, col, player):
        if col > 6:
            print("The colum does not exist!")
            return
        if self.board[0][col] != 0:
            #print("The column is full!")
            return
        if self.board[5][col] == 0:
            self.board[5][col] = player
        else:
            i = 0
            while(self.board[i][col] == 0):
                i += 1
            self.board[i-1][col] = player


    def reset(self):
        for i in range(6):
            self.board.append([0,0,0,0,0,0,0])

    # return true if game is over, otherwise return false
    def result(self):
        tie = True
        result = 0 #-1 means tie, 0 means undecided, 1 means player1 won, 2 means player2 won

        # streak resets every time a connection is broken. if streak == 4, there is a winner
        # Check Vertical
        for col in range(7):
            streak = 0
            for row in range(6):
                # This line checks if the board is full, thus tie.
                if self.board[row][col] == 0:
                    tie = False
                #If the cell is empty, then streak is 0
                if self.board[row][col] == 0:
                    streak = 0
                #if the cell is different from the previous cell, then streak is 1. if same, then streak += 1
                elif self.board[row][col] != self.board[row-1][col]:
                    streak = 1
                else:
                    streak +=1

                if streak == 4:
                    result = self.board[row][col]
        if tie:
            return -1

        # Check horizontal
        if(result == 0):
            for row in range(6):
                streak = 0
                for col in range(7):
                    if self.board[row][col] == 0:
                        streak = 0
                    elif self.board[row][col] != self.board[row][col-1]:
                        streak = 1
                    else:
                        streak +=1
                    if streak == 4:
                        result = self.board[row][col]


        # Check diagonal
        # each diagonal has a left top start point. then each cell is start_row + i, start_col + 1
        diagonals = [[2, 0, 4], [1, 0, 5], [0, 0, 6], [0, 1, 6], [0, 2, 5], [0, 3, 4]]
        for ind in range(6):
            if(result == 0):
                streak = 0
                row = diagonals[ind][0]
                col = diagonals[ind][1]
                for i in range(diagonals[ind][2]):
                    if self.board[row+i][col+i] == 0:
                        streak = 0
                    elif i != 0 and self.board[row+i][col+i] != self.board[row+i-1][col+i-1]:
                        streak = 1
                    else:
                        streak +=1
                    if streak == 4:
                        result = self.board[row+i][col-i]
        diagonals = [[2, 6, 4], [1, 6, 5], [0, 6, 6], [0, 5, 6], [0, 4, 5], [0, 3, 4]]
        for ind in range(6):
            if(result == 0):
                streak = 0
                row = diagonals[ind][0]
                col = diagonals[ind][1]
                for i in range(diagonals[ind][2]):
                    #print(i)
                    if self.board[row+i][col-i] == 0:
                        streak = 0
                    elif i != 0 and self.board[row+i][col-i] != self.board[row+i-1][col-i+1]:
                        streak = 1
                    else:
                        streak +=1
                    if streak == 4:
                        result = self.board[row+i][col-i]
                    #print("row: " + str(row+i) + ", col:" + str(col-i) + ", streak: " + str(streak))
        return result

    def start(self):
        player = 1
        print("Starting game...")
        self.display()
        while(self.result() == 0):
            message = "Player" + str(player)
            if player == 1:
                message += "(X)"
            else:
                message += "(O)"
            message += "'s turn\nEnter column index: "
            column = input(message)
            self.mark(int(column), player)
            if player == 1:
                player = 2
            else:
                player = 1
            self.display()
        if self.result() == -1:
            print("The board is full, game is Tie!")
        else:
            print("The winner is player " + str(self.result()))

    def get_state(self, player):
        if player == 1:
            return self.board, self.result()
        else:
            new_board = list()
            for i in range(len(self.board)):
                row = list()
                for j in range(len(self.board[i])):
                    if self.board[i][j] == 1:
                        row.append(2)
                    elif self.board[i][j] == 2:
                        row.append(1)
                    else:
                        row.append(0)
                new_board.append(row)
            return new_board, self.result()

    def input(self, action, player):
        self.mark(action, player)
