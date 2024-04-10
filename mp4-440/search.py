import heapq
# You do not need any other imports

def best_first_search(starting_state):
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''



    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search algorithm
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state) and obtain the path
    #   - keep track of the distance of each state from start node
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    # Your code here ---------------

    while (len(frontier) > 0):
        state_curr = heapq.heappop(frontier)
        if(state_curr.is_goal()):
            return backtrack(visited_states, state_curr) 
        neighbors_curr = state_curr.get_neighbors()
        # visited_stated[state] = {starting_state(frontier), (Parent, distance of each state from start node)}
        #visited[state[i]] = (Parent, distance of each state from start node)
        #visited[state[i]][0] = (Parent)
        #visited[state[i]][1] = (distance of each state from start node)
        for state in neighbors_curr:
            if (state not in visited_states) or state.dist_from_start < visited_states[state][1]: #If the state is not in visited states and also if the dist of that state is less than the dist of the state if it already exists
                heapq.heappush(frontier, state)
                visited_states[state] = (state_curr, state.dist_from_start)
    # ------------------------------
    # if you do not find the goal return an empty list
    return []

# TODO(III): implement backtrack method, to be called by best_first_search upon reaching goal_state
# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
def backtrack(visited_states, goal_state):
    path = []
    state_curr = goal_state
    # Your code here ---------------

    

    while (state_curr is not None):
        path.append(state_curr)
        state_curr = visited_states[state_curr][0]
    
    # ------------------------------
    path.reverse()
    return path