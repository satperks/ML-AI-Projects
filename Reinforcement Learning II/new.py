    def choose_action(self, s_prime):
        opt_action = utils.RIGHT
        food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right = s_prime
        opt_f = self.Q[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, opt_action]
        for action in [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP,  ]:
            if self._train:
                visit_times = self.N[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, action]
                if visit_times < self.Ne:
                    f = 1 
                else:
                    f = self.Q[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, action]
            else:
                f = self.Q[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, action]
            if f > opt_f:
                opt_action = action
                opt_f = f
        return opt_action
    def act(self, environment, points, dead):
        s_current = self.generate_state(environment)
        if self._train:  
            if self.s is not None and self.a is not None:
                if points == self.points + 1:
                    reward = 1
                elif dead:         
                    reward = -1
                else:
                    reward = -0.1
                self.update_n(self.s, self.a)
                self.update_q(self.s, self.a, reward, s_current)
            
            if dead:
                self.reset()
                action = np.random.choice(self.actions)
            else:
                self.s = s_current
                action = self.choose_action(s_current)
                self.a = action  
                self.points = points
        else:
            self.a = self.choose_action(s_current)
        return self.a
