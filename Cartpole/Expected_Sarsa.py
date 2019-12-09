import numpy as np
import math

class ExpectedSarsaAgent():
    
    def agent_init(self, agent_init_info):
        """
        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }
        
        """
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["random_seed"])
        
        # self.q = np.zeros((self.num_states, self.num_actions))
        self.q = np.zeros(self.num_states + (self.num_actions,))
        self.base_epsilon = self.epsilon
        self.base_step_size = self.step_size
        self.ada_divisor = 25
    
    def get_epsilon(self, t):
        return max(self.base_epsilon, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def get_alpha(self, t):
        return max(self.base_step_size, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))
    
    def agent_start(self, state):
        """
        Args:
            state (int): the state from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """
        current_q = self.q[state]
        # current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_step(self, reward, state):
        """
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """
        current_q = self.q[state]
        # current_q = self.q[state,:]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        count = 0
        for i in range(len(current_q)):
            if current_q[i] == np.max(current_q):
                count = count + 1
        
        expected = 0
        greedy = ((1-self.epsilon)/count) + (self.epsilon/self.num_actions)
        non_greedy = self.epsilon / self.num_actions

        for a in range(self.num_actions):
            if self.q[state][a] == np.max(current_q): expected = expected + greedy * self.q[state][a]
            else: expected = expected + non_greedy * self.q[state][a]

        
        expected = expected * self.discount
        self.q[self.prev_state][self.prev_action] = self.q[self.prev_state][self.prev_action] + \
        self.step_size * (reward + expected - self.q[self.prev_state][self.prev_action])
        
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_end(self, reward):
        """
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.q[self.prev_state][self.prev_action] = self.q[self.prev_state][self.prev_action] + \
        self.step_size * (reward - self.q[self.prev_state][self.prev_action])
        
    def argmax(self, q_values):
        """
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)
