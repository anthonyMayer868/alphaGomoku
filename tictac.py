import numpy
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import monte_carlo as mc 


'''
    Functions specific to the game which is a generalised version of tic tac toe.
    the size of the grid and the number needed in a row to win can be changed at
    the top of the script with s and w. 
    
    Player Class, contains value function, policy and function, and win statistics
                for the functions.
    Generation Class, contains best player,challenging player and the data generated 
                by their games. The play games method plays the games and returns the
                player with the most wins.
    Game class, Actally plays out the game between two players calling all of the game 
                and tree search function. contains data just for that game. 
                
                

'''




#size of board
s = 5
#number to win
w = 3



def soft_max(x):
    x = np.array(x)
    x = np.exp(x)
    x = x/np.sum(x)
    return x


def make_policy(l=7,board_size=s):
    '''
    creates players. l is the number of layers.

    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=2, activation='relu', padding='same' ,
                    input_shape=(2,board_size,board_size),bias_initializer=RandomUniform(seed=1)))
    for _ in range(l):
        model.add(Conv2D(64, kernel_size=2,padding='same' ,activation='relu',bias_initializer=RandomUniform()))

    model.add(Flatten())
    model.add(Dense(50,activation='relu',bias_initializer=RandomUniform()))
    model.add(Dense(1, activation='sigmoid',bias_initializer=RandomUniform()))
    model.compile(optimizer = 'adam', loss='mean_squared_error')
    return model

def make_value(l=5,board_size=s):
    '''
    creates players. l is the number of layers.
    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=2, activation='relu', padding='same' ,
                    input_shape=(1,board_size,board_size),bias_initializer=RandomUniform()))
    for _ in range(l):
        model.add(Conv2D(64, kernel_size=2,padding='same' ,activation='relu'))

    model.add(Flatten())
    model.add(Dense(50,activation='relu'))
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer = 'adam', loss='mean_squared_error')
    return model



def is_win(board,size=s,win_length=w):
    b = board
    size = size
    win_length = win_length
    for row in range(size):
        for col in range(size):
            try:
                dir1 = np.sum(b[row][col:col+win_length])
                if dir1 == win_length:
                    return 1
                if dir1 == -win_length:
                    return -1
            except:
                pass

            try:
                dir1 = np.sum(b[row][col-win_length+1:col+1])
                if dir1 == win_length:
                    return 1
                if dir1 == -win_length:
                    return -1
            except:
                pass

            try:
                dir1 = np.sum(b[:,col][row:row+win_length])
                if dir1 == win_length:
                    return 1
                if dir1 == -win_length:
                    return -1
            except:
                pass

            try:
                dir1 = np.sum(b[:,col][row-win_length+1:row])
                if dir1 == win_length:
                    return 1
                if dir1 == -win_length:
                    return -1
            except:
                pass
            diags = [(1,1),(-1,-1),(1,-1),(1,-1)]
            for d in diags:
                try:
                    dir1 = 0
                    for j in range(win_length):
                        if row + j*d[0]<0 or col +j*d[1]<0:
                            break
                        dir1 += b[row+j*d[0],col+j*d[1]]
                    if dir1 == win_length:
                        return 1
                    if dir1 == -win_length:
                        return -1
                except:
                    pass
    return 0

def is_terminal(board):
    if is_draw(board) or is_win(board) != 0:
        return True 
    return False 

def is_draw(board):
    if np.sum(board == 0) != 0:
        return 0
    return 1

def make_board(size=s):
    return np.zeros((s,s))

def make_move(b,move,turn):
    board = np.copy(b)
    if turn == 0:
        board[move[0],move[1]] = 1
    else:
        board[move[0],move[1]] = -1
    return board 


def legal_moves(board,player,size=s):
    moves = []
    for i in range(size):
        for j in range(size):
            if board[i,j] == 0:
                moves.append((i,j))
    if player == 'max':
        turn = 0
    else:
        turn = 1
    states = [make_move(board,m,turn) for m in moves]
    return states 


def create_value_vec(state):
    return state.reshape((1,s,s))

def create_action_vec(s1,s2):
    vec = np.zeros((2,s,s))
    vec[0] = s1 
    vec[1] = s2
    return vec

def value(model,state):
    vec = create_value_vec(state)
    return model.predict(np.array([vec]))

def policy(model,s1,s2):
    vec = create_action_vec(s1,s2)
    return model.predict(np.array([vec]))


def sigmoid(x):
    return 1/(1+np.exp(-np.array(x)))
  

class Game:

    def __init__(self,player1,player2):
        self.player1 = player1 
        self.player2 = player2 
        self.board = make_board()
        self.player1_tree = mc.Node(value,policy,player1,self.board,'max',legal_moves,is_terminal,[])
        self.player2_tree = None 
        self.x_val = np.zeros((s*s,1,s,s))
        self.y_val = np.ones((s*s,1))
        self.count = 1
        self.game_states = [self.board]
    def play_game(self):
        first_move = True
        while True:
            mc.tree_search(self.player1_tree,num_searches=50)
            values = [c.value/c.N for c in self.player1_tree.children]
            print([c.N for c in self.player1_tree.children])
            move = np.argmax(values)
            self.player1_tree = self.player1_tree.children[move]
            self.board = self.player1_tree.state
            self.game_states.append(self.board)
            self.x_val[self.count] = np.copy(create_value_vec(self.board))
            self.count += 1
            print(self.board,'\n')
            if is_win(self.board):
                for i in range(self.count):
                    self.y_val[i] = self.y_val[i]*(.9**(self.count-1-i))
                return 1 
            if is_draw(self.board):
                self.y_val = 0*self.y_val 
                return 0
                
            if first_move:
                self.player2_tree = mc.Node(value,policy,self.player2,self.board,'min',legal_moves,is_terminal,[])
                first_move = False 
            else:
                if self.player2_tree.children == []:
                    print('no children')
                    self.player2_tree.add_children()
  
                n = [i for i in range(len(self.player2_tree.children)) if np.array_equal(self.player2_tree.children[i].state,self.board)]
                
                self.player2_tree = self.player2_tree.children[n[0]]
            
            
            mc.tree_search(self.player2_tree,num_searches=50)
            values = [c.value/c.N for c in self.player2_tree.children]
            print([c.N for c in self.player2_tree.children])
            move = np.argmin(values)
            self.player2_tree = self.player2_tree.children[move]
            self.board = self.player2_tree.state
            self.game_states.append(self.board)
            print(self.board,'\n')
            self.x_val[self.count] = np.copy(create_value_vec(self.board))
            self.count += 1
            if is_win(self.board)==-1:
                self.y_val = -1*self.y_val
                for i in range(self.count):
                    self.y_val[i] = self.y_val[i]*(.9**(self.count-1-i))
                return -1
                 
            if is_draw(self.board):
                self.y_val = 0*self.y_val 
                return 0
            if self.player1_tree.children == []:
                    print('no children')
                    self.player1_tree.add_children()
  
            n = [i for i in range(len(self.player1_tree.children)) if np.array_equal(self.player1_tree.children[i].state,self.board)]
            
            self.player1_tree = self.player1_tree.children[n[0]]
        
    def clip_data(self):
        self.x_val = self.x_val[:self.count]
        self.y_val = self.y_val[:self.count]

    def get_action_data(self):
        x_action = []
        y_action = []
        
        for node in self.player1_tree.node_lst:
    
            if node.parent == None:
                pass 
            else:
                curr_node = node
                val = node.value/node.N 
                while curr_node.parent != None:
                    val_anc = curr_node.parent.value/curr_node.parent.N

                    if val > val_anc:
                        curr_node.action_data += 1
                    curr_node = curr_node.parent
        
        for node in self.player1_tree.node_lst:
        
            if node.parent == None:
                pass
            elif node.children == [] and node.terminal == False:
                pass 
            elif node.children == [] and node.terminal:
                win = is_win(node.state)
                if win == 1:
                    vec = create_action_vec(node.parent.state,node.state)
                    y_action.append(1)
                    x_action.append(vec)
                if win == -1:
                    vec = create_action_vec(node.parent.state,node.state)
                    y_action.append(-1)
                    x_action.append(vec) 
            else:
                vec = create_action_vec(node.parent.state,node.state)
                x_action.append(vec)
                y_action.append(node.action_data/node.N)
    
            
        
        for node in self.player2_tree.node_lst:
 
            if node.parent == None:
                pass 
            else:
                curr_node = node
                val = node.value/node.N 
                while curr_node.parent != None:
                    val_anc = curr_node.parent.value/curr_node.parent.N
                    if val > val_anc:
                        curr_node.action_data += 1
                    curr_node = curr_node.parent

        
        for node in self.player2_tree.node_lst:
            if node.parent == None:
                pass
            elif node.children == [] and node.terminal == False:
                pass 
            elif node.children == [] and node.terminal:
                win = is_win(node.state)
                if win == 1:
                    vec = create_action_vec(node.parent.state,node.state)
                    y_action.append(1)
                    x_action.append(vec)
                if win == -1:
                    vec = create_action_vec(node.parent.state,node.state)
                    y_action.append(-1)
                    x_action.append(vec) 
            else:
                vec = create_action_vec(node.parent.state,node.state)
                x_action.append(vec)
                y_action.append(node.action_data/node.N)
        return np.array(x_action),np.array(y_action)

class Player:

    def __init__(self):
        self.wins = 0 
        self.losses = 0
        self.models = [make_value(),make_policy()] 
    
    def train(self,vx,vy,ax,ay):
        self.models[0].fit(vx,vy,epochs=3,validation_split=.1)
        self.models[1].fit(ax,ay,epochs=10,validation_split = .1)


class Generation:

    def __init__(self,players,num_games=10):
        self.players = players
        self.action_data = np.zeros((350000,2,s,s))
        self.value_data = np.zeros((30000,1,s,s))
        self.action_tar = np.zeros((350000,1))
        self.value_tar = np.zeros((30000,1))
        self.action_count = 0
        self.value_count = 0
        self.num_games = num_games 
        self.sample_games = []
        
    def play_games(self):
        for i in range(self.num_games):
            #np.random.shuffle(self.players)
            player1 = self.players[0].models
            player2 = self.players[1].models
            g = Game(player1,player2)
            win = g.play_game()
            if win:
                self.players[0].wins += 1
            if win == -1:
                self.players[1].wins += 1
            a_x,a_y = g.get_action_data()
            g.clip_data()
            v_x,v_y = g.x_val,g.y_val
            n = len(v_x)
            m = len(a_x)
            v_y = v_y.reshape(n,1)
            a_y = a_y.reshape(m,1)
            self.action_data[self.action_count:self.action_count+m] = a_x 
            self.action_tar[self.action_count:self.action_count+m] = a_y
            self.value_data[self.value_count:self.value_count+n] = v_x 
            self.value_tar[self.value_count:self.value_count+n] = v_y  
            self.action_count += m 
            self.value_count += n
            self.sample_games.append(g.game_states) 
            print('Game {} Done {} {}'.format(i,n,m))
        for i in range(self.num_games):
            #np.random.shuffle(self.players)
            player1 = self.players[1].models
            player2 = self.players[0].models
            g = Game(player1,player2)
            win = g.play_game()
            if win:
                self.players[1].wins += 1
            if win == -1:
                self.players[0].wins += 1
            a_x,a_y = g.get_action_data()
            g.clip_data()
            v_x,v_y = g.x_val,g.y_val
            n = len(v_x)
            m = len(a_x)
            v_y = v_y.reshape(n,1)
            a_y = a_y.reshape(m,1)
            self.action_data[self.action_count:self.action_count+m] = a_x 
            self.action_tar[self.action_count:self.action_count+m] = a_y
            self.value_data[self.value_count:self.value_count+n] = v_x 
            self.value_tar[self.value_count:self.value_count+n] = v_y  
            self.action_count += m 
            self.value_count += n
            self.sample_games.append(g.game_states) 
            print('Game {} Done {} {}'.format(i,n,m))
        if self.players[0].wins>= self.players[1].wins:
            return self.players[0] 
        else:
            print('best player change')
            return self.players[1]


    def clip_data(self):
        '''
            Space in pre-allocated so this removes 
            unused rows.
        '''
        self.value_data = self.value_data[:self.value_count]
        self.action_data = self.action_data[:self.action_count]
        self.value_tar = self.value_tar[:self.value_count]
        self.action_tar = self.action_tar[:self.action_count]
    
    

def augment_v(vx,vy):
    '''
        The game is rotationaly invarient so we exploit this to
        increase the training data.
    
    '''
    n = vx.shape[0]
    A = np.zeros((4*vx.shape[0],1,s,s))
    Ay = np.zeros((4*vx.shape[0],1))
    for i in range(n):
        vec = vx[i][0]
        tar = vy[i][0]
        vec1 = np.rot90(vec, k=1)
        vec2 = np.rot90(vec, k=2)
        vec3 = np.rot90(vec, k=3)
        A[4*i] = vec.reshape(1,s,s) 
        A[4*i+1] = vec1.reshape(1,s,s)
        A[4*i+2] = vec2.reshape(1,s,s)
        A[4*i+3] = vec3.reshape(1,s,s)
        Ay[4*i] = tar 
        Ay[4*i+1] = tar
        Ay[4*i+2] = tar 
        Ay[4*i+3] = tar 
    return A,Ay  






def play_random(player1):
    '''
       It can  be hard to tell the difference
       between improvement and randomness so
       a player can be made to play against
       random moves as a basic objective 
       measure.
    '''
    board = make_board()
    player1_tree = mc.Node(value,policy,player1,board,'max',legal_moves,is_terminal,[])
    while True:
        mc.tree_search(player1_tree,num_searches=35)
        values = [c.value/c.N for c in player1_tree.children]
        move = np.argmax(values)
        player1_tree = player1_tree.children[move]
        board = player1_tree.state
        print('')
        print(board)
        print('')
        if is_win(board):
            print('ai win')
            break
        if is_draw(board):
            print('draw')
            break

        
        moves = legal_moves(board,'min')
        np.random.shuffle(moves)
        board = moves[0]
        print(board)
        if is_win(board)==-1:
            print('random wins')
            break 
        if is_draw(board):
            print('draw')
            break
            
        n = [i for i in range(len(player1_tree.children)) if np.array_equal(player1_tree.children[i].state,board)]
        
        print(len(player1_tree.children))
        player1_tree = player1_tree.children[n[0]]



