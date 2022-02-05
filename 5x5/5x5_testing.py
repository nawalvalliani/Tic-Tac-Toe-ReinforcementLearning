import numpy as np
import pickle
import copy

global_rows = 5
global_columns = global_rows

global_win = 4

outfile = open("moves_.txt", "w")

outfile.write("5x5 Tic Tac Toe - Human vs. AI\n")
print("5x5 Tic Tac Toe - Human vs. AI")

# State Represenation
class State:
    def __init__(self, player1, player2, rows=global_rows, columns=global_columns):
        self.rows = rows
        self.columns = columns
        self.board = np.zeros((self.rows, self.columns))
        self.board_array = None
        self.player1 = player1
        self.player2 = player2
        self.done = False

        # The first player will be represented as 1 and Player 2 will have -1 representation

        self.player_mark = 1

    def showBoard(self):
        print("    0   1   2   3   4")
        for i in range(self.rows):
            print('  ----------------------')
            out = '{0} | '.format(i)
            for j in range(0, self.columns):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('  ----------------------')

    def convertBoardToArray(self):
        self.board_array = str(self.board.reshape(self.rows * self.columns))
        return self.board_array

    def availablePosition(self):
        a = []
        for i in range(self.rows):
            for j in range(self.columns):
                if self.board[i, j] == 0:
                    a.append((i, j))
        return a

    def check_win(self, array):
        if len(array) > 4:
            if array[0] == array[1] == array[2] == array[3] == 1:
                self.done = True
                return 1
            elif array[1] == array[2] == array[3] == array[4] == 1:
                self.done = True
                return 1
            if array[0] == array[1] == array[2] == array[3] == -1:
                self.done = True
                return -1
            elif array[1] == array[2] == array[3] == array[4] == -1:
                self.done = True
                return -1
        else:
            if array[0] == array[1] == array[2] == array[3] == 1:
                self.done = True
                return 1
            if array[0] == array[1] == array[2] == array[3] == -1:
                self.done = True
                return -1

    def Win(self):
        for j in range(0, global_columns):
            ones = self.check_win(self.board[:, j])
            if ones is not None:
                return ones
        for i in range(0, global_rows):
            ones = self.check_win(self.board[i, :])
            if ones is not None:
                return ones
        x = copy.deepcopy(self.board)
        diags = [x[::-1, :].diagonal(i) for i in range(-x.shape[0] + 1, x.shape[1])]
        diags.extend(x.diagonal(i) for i in range(x.shape[1] - 1, -x.shape[0], -1))
        for each in diags:
            if len(each) >= 4:
                ones = self.check_win(each)
                if ones is not None:
                    return ones
        isTie = self.Tie()
        if isTie == 0:
            return isTie
        self.done = False
        return None

    '''def Win(self):

        for i in range(self.rows):
            if sum(self.board[i, :]) == global_win:
                self.done = True
                return 1
            if sum(self.board[i, :]) == -1*global_win:
                self.done = True
                return -1
        for j in range(self.columns):
            if sum(self.board[:, j]) == global_win:
                self.done = True
                return 1
            if sum(self.board[:, j]) == -1*global_win:
                self.done = True
                return -1
        # Diagonals
        diagonal1 = sum([self.board[i, i] for i in range(self.columns)])
        diagonal2 = sum([self.board[i, self.columns - i - 1] for i in range(self.columns)])
        if diagonal1 == global_win or diagonal2 == global_win:
            self.done = True
            return 1
        if diagonal1 == -1*global_win or diagonal2 == -1*global_win:
            self.done = True
            return -1

        isTie = self.Tie()
        if isTie == 0:
            return isTie
        self.done = False
        return None'''

    def Tie(self):
        if (len(self.availablePosition())) == 0:
            self.done = True
            return 0

    def ToggleSymbol(self):
        self.player_mark = -self.player_mark

    def updateState(self, position):
        self.board[position] = self.player_mark
        if self.player_mark == 1:
            self.player_mark = -1
        else:
            self.player_mark = 1

    def updateState_play(self, position):
        self.board[position] = self.player_mark
        outfile.write("AI Move: {0}\n".format(position))
        if self.player_mark == 1:
            self.player_mark = -1
        else:
            self.player_mark = 1

    def reward(self):
        win = self.Win()
        if win == 1:
            self.player1.giveReward(1)
            self.player2.giveReward(-1)
        elif win == -1:
            self.player1.giveReward(-1)
            self.player2.giveReward(1)
        else:
            self.player1.giveReward(0.1)
            self.player2.giveReward(0.5)

    def train(self, number_of_rounds=100000, modulo=10000):
        print("Total training rounds: ", number_of_rounds)
        for i in range(number_of_rounds):
            if i % modulo == 0:
                print("Round number:", i)
            while not self.done:
                if np.random.uniform(0, 1) <= 0.1:
                    self.ToggleSymbol()
                else:
                    available_positions = self.availablePosition()
                    action1 = self.player1.selectAction(available_positions, self.board, self.player_mark)
                    self.updateState(action1)
                    boardArray = self.convertBoardToArray()
                    self.player1.addState(boardArray)
                    win = self.Win()
                    if win is not None:
                        self.reward()
                        self.player1.reset()
                        self.player2.reset()
                        self.reset()
                        break
                if np.random.uniform(0, 1) <= 0.1:
                    self.ToggleSymbol()
                else:
                    available_positions = self.availablePosition()
                    action2 = self.player2.selectAction(available_positions, self.board, self.player_mark)
                    self.updateState(action2)
                    boardArray1 = self.convertBoardToArray()
                    self.player2.addState(boardArray1)
                    win = self.Win()
                    if win is not None:
                        self.reward()
                        self.player1.reset()
                        self.player2.reset()
                        self.reset()
                        break

    def playwithHuman(self):
        while not self.done:
            available_positions = self.availablePosition()
            action1 = self.player1.selectAction(available_positions, self.board, self.player_mark)
            #self.updateState(action1)
            self.updateState_play(action1)
            boardArray = self.convertBoardToArray()
            #             self.player1.addState(boardArray)
            self.showBoard()
            win = self.Win()
            if win is not None:
                if win == 1:
                    print(self.player1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break
            else:
                available_positions = self.availablePosition()
                action2 = self.player2.selectAction(available_positions)
                self.updateState(action2)
                boardArray1 = self.convertBoardToArray()
                #                 self.player2.addState(boardArray1)
                self.showBoard()
                win = self.Win()
                if win is not None:
                    if win == -1:
                        print(self.player2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def reset(self):
        self.board = np.zeros((self.rows, self.columns))
        self.board_array = None
        self.done = False
        self.player_mark = 1


class Agent():
    def __init__(self, name, epsilon=0.3, rows=global_rows, columns=global_columns):
        self.name = name
        self.rows = rows
        self.columns = columns
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = epsilon
        self.states_value = {}
        self.state_list = []
        self.sum_of_qvalues = []

    def convertBoardToArray(self, board):
        board_array = str(board.reshape(self.rows * self.columns))
        return board_array

    def addState(self, state):
        self.state_list.append(state)

    def selectAction(self, position_available, board, player_mark):
        if np.random.uniform(0, 1) <= self.epsilon:
            index = np.random.choice(len(position_available))
            action = position_available[index]
        else:
            value_max = -999
            for p in position_available:
                next_board = board.copy()
                next_board[p] = player_mark
                next_board_Array = self.convertBoardToArray(next_board)
                if self.states_value.get(next_board_Array) is None:
                    value = 0
                else:
                    value = self.states_value.get(next_board_Array)

                if value >= value_max:
                    value_max = value
                    action = p
        return action

    def giveReward(self, reward):
        for i in reversed(self.state_list):
            if self.states_value.get(i) is None:
                self.states_value[i] = 0
            self.states_value[i] += self.alpha * (self.gamma * reward - self.states_value[i])
            reward = self.states_value[i]
        self.sum_of_qvalues.append(sum(self.states_value.values()))

    def reset(self):
        self.state_list = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class Human:
    def __init__(self, name):
        self.name = name

    def selectAction(self, position):
        while True:
            #r = int(input("Input your action row(range:0 to 2)"))
            #c = int(input("Input your action column(range:0 to 2):"))
            r = int(input("Input your action row(range: 0 to {0}): ".format(global_rows-1)))
            c = int(input("Input your action column(range: 0 to {0}): ".format(global_columns-1)))
            action = (r, c)
            outfile.write("Human Move: {0}\n".format(action))
            if action in position:
                return action


#p1 = Agent("p1")
#st = State(p1, p1)
#print("Training...")
training_rounds = 100000
#outfile.write("Number of training rounds: {0}".format(training_rounds))
#st.train(number_of_rounds=training_rounds,modulo=100)

#p1.savePolicy()
#p1.loadPolicy("policy_p1_10000")
p1 = Agent("computer", epsilon=0)
p1.loadPolicy("policy_p1_10000")

p2 = Human("human")

st = State(p1, p2)
st.playwithHuman()

outfile.close()
