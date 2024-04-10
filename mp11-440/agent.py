import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
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
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table. 
        finIndx = tuple(state + (action,))
        self.N[finIndx] += 1

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table. 
        p_s, state = s, s_prime
        p_i = tuple(p_s + (a,))
        N = self.N[p_i]
        learning_rate = self.C / (self.C + N) 
        pri = self.Q[p_i]
        maxed = max(self.Q[state + (action,)] for action in self.actions)
        final_updated = pri + learning_rate * (r + self.gamma * maxed - pri)
        self.Q[p_i] = final_updated

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO - MP12: write your function here

        return utils.RIGHT
    
    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment

        shx, shy = environment[0], environment[1]
        sb = environment[2]
        fx, fy = environment[3], environment[4]
        rx, ry = environment[5], environment[6]

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

        x_wall = 0
        if(shx==1):
            x_wall = 1
        elif(shx==self.display_width-2):
            x_wall = 2

        y_wall = 0
        if(shy==1):
            y_wall = 1
        elif(shy==self.display_height-2):
            y_wall = 2

        b_top = 0
        b_bottom = 0
        b_left = 0
        b_right = 0

        if shx == 1 or (shx == rx + 2 and shy == ry): 
            x_wall = 1
        elif shx == self.display_width - 2 or (shx == rx - 1 and shy == ry):
            x_wall = 2
        else:
            x_wall = 0
        if shy == 1 or ((shx == rx  or shx == rx + 1) and shy == ry + 1): 
            y_wall = 1
        elif shy == self.display_height - 2 or ((shx == rx  or shx == rx + 1) and shy == ry - 1):
            y_wall = 2
        else:
            y_wall = 0

        for pos in sb:
            if(shx == pos[0] and (shy-1)==pos[1]):
                b_top = 1

        for pos in sb:
            if(shx == pos[0] and (shy+1)==pos[1]):
               b_bottom = 1

        for pos in sb:
            if((shx-1) == pos[0] and shy==pos[1]):
                b_left = 1

        for pos in sb:
            if((shx+1) == pos[0] and shy==pos[1]):
                b_right = 1
        return (food_dir_x, food_dir_y, x_wall, y_wall, b_top, 
        b_bottom, b_left, b_right)