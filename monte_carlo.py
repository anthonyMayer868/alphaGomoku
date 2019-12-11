import numpy as np

'''
GOAL: make a general purpose monte carlo tree search class. 
      to be used for an assortment of games.

    The general algorithm:
        Value function trained to predict winner of the 
        results of the played games.

        pol function tries to predict the results of monte carlo tree
        search. specifically tries to predict the probability 
        that an action will have child states with higher value then
        the current state. 

'''

def soft_max(x):
    x = np.array(x)
    x = np.exp(x)
    x = x/np.sum(x)
    return x

class Node:

    def __init__(self,val,pol,models,state,player,find_children,is_terminal,node_lst,parent = None):
        '''
            val = value function. val(state,player) returns value on [-1,1]

            pol = probability val increases because of an action. 
                  pol(s1,s2) return value on [0,1]
            
            state = state of game at the node 
            
            player = minimizer of maximizer of the value function
            
            find_children game dependent function that returns all possible child states
            
            is_terminal is a game dependent function that checks for wins or draws

            node_lst a list of all the nodes for easy extraction of data. 
        '''
        
        node_lst.append(self)
        self.node_lst = node_lst
        self.player = player
        self.parent = parent 
        self.models = models
        self.val = val 
        self.pol = pol 
        self.state = state
        self.is_terminal = is_terminal
        self.terminal = is_terminal(state)
        self.find_children = find_children
        self.children = [] 
        self.N = 1
        self.n = 1
        self.action_data = 0
        self.dont_return = False #in case only leads to terminal states
        if self.terminal:
            self.leaf = False 
        else:
            self.leaf = True
        if parent != None:
            self.prob = pol(models[1],self.parent.state,state)[0][0]
            self.value = val(models[0],state)[0][0]
        else:
            self.prob = None
            self.value = val(models[0],state)[0][0]
    
    
    def add_children(self):
        if self.player == 'max':
            p = 'min'
        else:
            p  = 'max'
        self.leaf = False
        child_states = self.find_children(self.state,self.player)
        self.children = [Node(self.val,self.pol,self.models,child,p,self.find_children,self.is_terminal,self.node_lst,parent = self) for child in child_states]
        for child in self.children:
            curr_node = child
            v = curr_node.value
            p = curr_node.prob
            while curr_node.parent.prob != None:
                curr_node.parent.value += v 
                curr_node.parent.prob += p 
                curr_node.parent.N += 1
                curr_node = curr_node.parent

def roll_out(node,orig_node,a = .01):
    global depth
    if depth > 800:
        return
    if node.leaf:
        node.add_children()
    elif all([c.dont_return for c in orig_node.children]):
        pass
    else:
        if node.player == 'max':
            q_vals = []
            for c in node.children:
                if c.terminal or c.dont_return:
                    q_vals.append(-10000)
                else:
                    q_vals.append(c.value/c.N + c.prob/c.N)
            if all([q==-10000 for q in q_vals]):
                node.dont_return == True 
                depth += 1
                roll_out(orig_node,orig_node,a = a)
            else:
                q_vals = soft_max(q_vals)
                moves = [n for n in range(len(q_vals))]
                move = np.random.choice(moves,p=q_vals)
                node = node.children[move]
                node.n += 1
                depth += 1
                roll_out(node,orig_node,a=a)
        else:
            q_vals = []
            for c in node.children:
                if c.terminal or c.dont_return:
                    q_vals.append(-10000)
                else:
                    q_vals.append(-c.value/c.N + 1-c.prob/c.N)
            if all([q==-10000 for q in q_vals]):
                node.dont_return == True
                depth += 1 
                roll_out(orig_node,orig_node,a = a)
            else:
                q_vals = soft_max(q_vals)
                moves = [n for n in range(len(q_vals))]
                move = np.random.choice(moves,p=q_vals)
                node = node.children[move]
                node.n += 1
                depth += 1
                roll_out(node,orig_node,a=a)

depth = 0 #need to keep track of recursion depth                           
found_leaf = True #keeps from over counting the number
                  #times a node was visited
def tree_search(node,num_searches=1000):
    global depth 

    for _ in range(num_searches):
        depth = 0
        roll_out(node,node)
    visits = [c.n for c in node.children]
    return visits


