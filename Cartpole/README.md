# Cartpole implementation
## Content
[Dyna-Q+ algorithm](#dyna-q-algorithm)  
- [Parameter Setting](#parameter-setting)  
- [Results](#results)  

[Expected Sarsa algorithm](#expected-sarsa-algorithm)  
- [Parameter Setting](#parameter-setting-1)  
- [Results](#results-1)  

[Method Comparison](#method-comparison)  
[Note](#note)  
[References](#references)  


## Dyna-Q+ algorithm
### Parameter Setting
The agent(Dyna-Q+) parameter is being set as follows:  
* epsilon = 0.1  
* step size(lr) = 0.01  
* gamma(discount) = 1  
* kappa = 0.001  
* planning steps = [0, 5, 10, 50]
* num of episode = 500

### Results
#### Without adaptive learning
![alt text](https://github.com/yanshuolee/RL-implementation/blob/master/Cartpole/DynaQ_plus_results/step_size_0.01/various_planning_steps.png)
X axis: episode  
Y axis: reward  
The max score is 146.928 on average 500 episode with planning step of 50.

#### With adaptive learning
![alt text]()
X axis: episode  
Y axis: reward  
The max score is  on average 500 episode with planning step of .

## Expected Sarsa algorithm
### Parameter Setting
The agent(Expected Sarsa) parameter is being set as follows:  
* epsilon = 0.1  
* step size(lr) = [0.01, 0.05, 0.1, 0.5]  
* gamma(discount) = 1    
* num of episode = 500

### Results
#### Without adaptive learning
![alt text](https://github.com/yanshuolee/RL-implementation/blob/master/Cartpole/Expected_Sarsa_results/various_step_size.png)
X axis: episode  
Y axis: reward  
The max score is 93.38 on average 500 episode with step size of 0.1.

#### With adaptive learning
![alt text]()
X axis: episode  
Y axis: reward  
The max score is  on average 500 episode with planning step of .

## Method Comparison
Cartpole game is defined as being solved as getting avg reward of 195 over 100 consecutive trials.
* [Q-learning: score = 200](https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947)
* [A3C: scores >> 200](https://medium.com/tensorflow/deep-reinforcement-learning-playing-cartpole-through-asynchronous-advantage-actor-critic-a3c-7eab2eea5296)

## Note
The cartpole score using Q-learning method [1] converges to 200 rewards after some episodes while in A3C [2] it converges to more than 300 rewards. My implementation using Dyna-Q+ with optimal parameter setting reaches average reward of 146.928 on 500 episodes. However, the plots shows that using Dyna-Q+ fluctuates drastically.  

## References
[1] https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947  
[2] https://gist.github.com/n1try/af0b8476ae4106ec098fea1dfe57f578  
[3] http://www.derongliu.org/adp/adp-cdrom/Barto1983.pdf
