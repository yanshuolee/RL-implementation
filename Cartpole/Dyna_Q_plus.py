import numpy as np

class DynaQPlusAgent():
    
    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts.

        Args:
            agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
            {
                num_states (int): The number of states,
                num_actions (int): The number of actions,
                epsilon (float): The parameter for epsilon-greedy exploration,
                step_size (float): The step-size,
                discount (float): The discount factor,
                planning_steps (int): The number of planning steps per environmental interaction
                kappa (float): The scaling factor for the reward bonus

                random_seed (int): the seed for the RNG used in epsilon-greedy
                planning_random_seed (int): the seed for the RNG used in the planner
            }
        """

        try:
            self.num_states = agent_info["num_states"]
            self.num_actions = agent_info["num_actions"]
        except:
            print("You need to pass both 'num_states' and 'num_actions' \
                   in agent_info to initialize the action-value table")
        self.gamma = agent_info.get("discount", 0.95)
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self.planning_steps = agent_info.get("planning_steps", 10)
        self.kappa = agent_info.get("kappa", 0.001)

        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 42))
        self.planning_rand_generator = np.random.RandomState(agent_info.get('planning_random_seed', 42))

        # self.q_values = np.zeros((self.num_states, self.num_actions))
        # self.tau = np.zeros((self.num_states, self.num_actions))

        self.q_values = np.zeros(self.num_states + (self.num_actions,))
        self.tau = np.zeros(self.num_states + (self.num_actions,))
        self.actions = list(range(self.num_actions))
        self.past_action = -1
        self.past_state = -1
        self.model = {}

    def update_model(self, past_state, past_action, state, reward):
        """updates the model 

        Args:
            past_state  (int): s
            past_action (int): a
            state       (int): s'
            reward      (int): r
        Returns:
            Nothing
        """

        if past_state not in self.model:
            self.model[past_state] = {past_action : (state, reward)}

            for action in self.actions:
                if action != past_action:
                    self.model[past_state][action] = (past_state, 0)
        else:
            self.model[past_state][past_action] = (state, reward)
    
    def randtuple(self, tuples_in_list):
        index = self.planning_rand_generator.choice(np.arange(len(tuples_in_list)))
        return tuples_in_list[index]

    def planning_step(self):
        """performs planning, i.e. indirect RL.

        Args:
            None
        Returns:
            Nothing
        """

        for _ in range(self.planning_steps):
            pt_state = self.randtuple(list(self.model.keys()))
            pt_action = self.planning_rand_generator.choice(list(self.model[pt_state].keys()))
            s_prime, R = self.model[pt_state][pt_action]
            R = R + self.kappa * (self.tau[pt_state][pt_action])**(1/2.0)
            if s_prime != -1:
                self.q_values[pt_state][pt_action] = self.q_values[pt_state][pt_action] + \
                self.step_size * (R + self.gamma * np.max(self.q_values[s_prime]) - self.q_values[pt_state][pt_action])
            else:
                self.q_values[pt_state][pt_action] = self.q_values[pt_state][pt_action] + \
                self.step_size * (R - self.q_values[pt_state][pt_action])
    
    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action values
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

    def choose_action_egreedy(self, state):
        """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.

        Args:
            state (List): coordinates of the agent (two elements)
        Returns:
            The action taken w.r.t. the aforementioned epsilon-greedy policy
        """

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.choice(self.actions)
        else:
            values = self.q_values[state]
            action = self.argmax(values)

        return action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) The first action the agent takes.
        """

        self.past_state = state
        self.past_action = self.choose_action_egreedy(self.past_state) 
        
        return self.past_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent is taking.
        """  

        self.tau += 1
        self.tau[self.past_state][self.past_action] = 0
        self.q_values[self.past_state][self.past_action] = self.q_values[self.past_state][self.past_action] + \
        self.step_size * (reward + self.gamma * np.max(self.q_values[state]) - self.q_values[self.past_state][self.past_action])
        self.update_model(self.past_state, self.past_action, state, reward)
        self.planning_step()
        action = self.choose_action_egreedy(state)
        self.past_action = action
        self.past_state = state
        
        return self.past_action

    def agent_end(self, reward):
        """Called when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        self.tau += 1
        self.tau[self.past_state][self.past_action] = 0
        self.q_values[self.past_state][self.past_action] = self.q_values[self.past_state][self.past_action] + \
        self.step_size * (reward - self.q_values[self.past_state][self.past_action])
        self.update_model(self.past_state, self.past_action, -1, reward)
        self.planning_step()
