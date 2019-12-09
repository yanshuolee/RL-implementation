import gym
import math
import numpy as np
# import Expected_Sarsa as Agent
import Dyna_Q_plus as Agent

num_episodes = 500
buckets=(1, 1, 6, 12,)
agent_info = {"num_actions": 2, 
              "num_states": buckets, 
              "epsilon": 0.1, 
              "step_size": 0.01, 
              "discount": 1.0,
              "kappa": 0.001,
              "planning_steps": 5,
              "random_seed": 0,
              "planning_random_seed": 0}

# agent = Agent.ExpectedSarsaAgent()
agent = Agent.DynaQPlusAgent()

def discretize(obs, env):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

if __name__ == "__main__":
    # step_size = [0.01, 0.05, 0.1, 0.5]
    planning_steps = [0, 5, 10, 50]

    for step in planning_steps:
        agent_info["planning_steps"] = step
        agent.agent_init(agent_info)
        env = gym.make("CartPole-v1")
        Rewards = []
        
        for ep in range(num_episodes):
            total_rewards = 0
            last_state = discretize(env.reset(), env)
            done = False
            last_action = agent.agent_start(last_state)

            agent.epsilon = agent.get_epsilon(ep)
            agent.step_size = agent.get_alpha(ep)
            
            count_steps = 0
            while not done:
                count_steps += 1
                obs, reward, done, _ = env.step(last_action)
                total_rewards += reward
                last_state = discretize(obs, env)
                last_action = agent.agent_step(reward, last_state)
            
            print("Episode: {} with {} planning step(s) Total reward: {}".format(ep, agent.planning_steps, total_rewards))
            Rewards.append(total_rewards)
        
        np.save("./DynaQ_plus_results/step_size_0.01/adaptive/planning_step_{}".format(agent.planning_steps), Rewards)
        