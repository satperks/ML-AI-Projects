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

    def update_q(self, s, a, r, fin):
        # TODO - MP11: Update the Q-table. 
        p_s, state = s, fin
        p_i = tuple(p_s + (a,))
        N = self.N[p_i]
        learning_rate = self.C / (self.C + N) 
        pri = self.Q[p_i]
        maxed = max(self.Q[state + (action,)] for action in self.actions)
        final_updated = pri + learning_rate * (r + self.gamma * maxed - pri)
        self.Q[p_i] = final_updated

    

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
    
#-------------------------------------------------------------------------------------------------------------------------------------------
    
    def next_move(self, s_prime):
        o_action = utils.RIGHT
        fin = self.calculate_optimal_f(s_prime, o_action)

        for action in [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP]:
            f = self.evaluate_action_f(s_prime, action)
            if f > fin:
                o_action, fin = action, f

        return o_action

    def evaluate_action_f(self, s_prime, action):
        if self._train:
            v = self.N[s_prime + (action,)]
            return 1 if v < self.Ne else self.Q[s_prime + (action,)]
        return self.Q[s_prime + (action,)]

    def calculate_optimal_f(self, s_prime, action):
        return self.Q[s_prime + (action,)]

    def act(self, environment, points, dead):
        curr = self.generate_state(environment)
        if self._train:
            self.process_training(points, dead, curr)
            action = self.take_training_action(points, dead, curr)
        else:
            action = self.next_move(curr)

        return action

    def process_training(self, points, dead, curr):
        if self.s is not None and self.a is not None:
            r = self.calculate_reward(points, dead)
            self.update_q_and_n(r, curr)

    def calculate_reward(self, points, dead):
        if points == self.points + 1:
            return 1
        if dead:
            return -1
        return -0.1

    def update_q_and_n(self, r, curr):
        self.update_n(self.s, self.a)
        self.update_q(self.s, self.a, r, curr)

    def take_training_action(self, points, dead, curr):
        if dead:
            self.reset()
            return np.random.choice(self.actions)
        self.s, self.a = curr, self.next_move(curr)
        self.points = points
        return self.a