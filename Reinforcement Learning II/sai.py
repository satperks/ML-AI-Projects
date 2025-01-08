import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    # def optimalAction(self):
    #     for a in self.actions:


    #     return utils.RIGHT


    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        state = self.generate_state(environment) 
        if(self._train and self.s!= None and self.a!=None):
            reward = -0.1
            if(points>self.points):
                reward = 1
            if(dead):
                reward = -1
        
            self.N[self.s][self.a] += 1
            action = np.max(self.Q[state])
            alpha = self.C/(self.C+self.N[self.s][self.a])
            self.Q[self.s][self.a] = self.Q[self.s][self.a] + alpha*(reward + self.gamma * action - self.Q[self.s][self.a])
            if(dead):
                self.reset()
                return utils.RIGHT
    
        maxQ = -99999999
        optAction = -1
        for a in self.actions:
            if(self._train and self.N[state][a]<self.Ne):
                optAction = a
                maxQ = 1
            elif(maxQ<=self.Q[state][a]):
                optAction = a
                maxQ = self.Q[state][a]
        self.s = state
        self.a = optAction
        self.points = points
        return optAction
    
    def act(self, environment, points, dead):
        s_prime = self.generate_state(environment)
        if (self._train and self.s is not None and self.a is not None): 
            self.N[self.s][self.a]+=1

            alpha = self.C/(self.C+self.N[self.s][self.a])

            if dead:
                r = -1
            # elif (points-self.points>0):
            elif (points == self.points+1):
                r = 1
            else:
                r = -0.1

            old_value = self.Q[self.s][self.a]
            
            maxQ = np.max(self.Q[s_prime])
            self.Q[self.s][self.a] = old_value + alpha * (r + self.gamma * maxQ - old_value)   
        if dead: 
            self.reset()
            return 0
        
        maxQ = -99999999
        optAction = -1
        for a in self.actions:
            if(self._train and self.N[s_prime][a]<self.Ne):
                optAction = a
                maxQ = 1
            elif(maxQ<=self.Q[s_prime][a]):
                optAction = a
                maxQ = self.Q[s_prime][a]
        self.s = s_prime
        self.a = optAction
        self.points = points
        return optAction
        

        a_prime = 3
        a_prime_val = self.Q[s_prime][3]

        def exploreCondition(qval, nval):
            if self._train and nval < self.Ne:
                return 1
            else:
                return qval

        for a_option in self.actions: # up, down, left and right
            val = exploreCondition(self.Q[s_prime][a_option], self.N[s_prime][a_option]) 
            if a_prime_val < val:
                a_prime = a_option
                a_prime_val = val 

        self.s = s_prime 
        self.a = a_prime
        self.points = points
        return a_prime
    

    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        shx, shy, sb, fx, fy = environment

        food_dir_x = 0
        if(fx < shx):
            food_dir_x = 1
        elif(fx > shx):
            food_dir_x = 2
        
        food_dir_y = 0
        if(fy > shy):
            food_dir_y = 2
        elif(fy < shy):
            food_dir_y = 1

        adjoining_wall_x = 0
        if(shx==1):
            adjoining_wall_x = 1
        elif(shx==utils.DISPLAY_WIDTH-2):
            adjoining_wall_x = 2

        adjoining_wall_y = 0
        if(shy==1):
            adjoining_wall_y = 1
        elif(shy==utils.DISPLAY_HEIGHT-2):
            adjoining_wall_y = 2
        #check for case if snake outside??

        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0

        #check top
        for pos in sb:
            if(shx == pos[0] and (shy-1)==pos[1]):
                adjoining_body_top = 1
        #check bottom
        for pos in sb:
            if(shx == pos[0] and (shy+1)==pos[1]):
                adjoining_body_bottom = 1
        #check left
        for pos in sb:
            if((shx-1) == pos[0] and shy==pos[1]):
                adjoining_body_left = 1
        #check right
        for pos in sb:
            if((shx+1) == pos[0] and shy==pos[1]):
                adjoining_body_right = 1
        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, 
        adjoining_body_bottom, adjoining_body_left, adjoining_body_right)