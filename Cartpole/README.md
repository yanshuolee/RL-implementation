# Cartpole implementation
## Results using Dyna-Q+ algorithm
### Parameter Setting
The agent(Dyna-Q+) parameter is being set as follows:  
* epsilon = 0.1  
* step size(lr) = [0.01, 0.05]  
* gamma(discount) = 1  
* kappa = 0.001  
* planning steps = [0, 5, 10, 50]
* num of episode = 500

### Results
#### Using step size of 0.01
![alt text](https://github.com/yanshuolee/RL-implementation/blob/master/Cartpole/results/step_size_0.01/various_planning_steps.png)

#### Using step size of 0.05
![alt text](https://github.com/yanshuolee/RL-implementation/blob/master/Cartpole/results/step_size_0.05/various_planning_steps.png)

## Method Comparison
Cartpole game is defined as being solved as getting avg reward of 195 over 100 consecutive trials.
* [Q-learning: score = 200](https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947)
* [A3C: scores >> 200](https://medium.com/tensorflow/deep-reinforcement-learning-playing-cartpole-through-asynchronous-advantage-actor-critic-a3c-7eab2eea5296)

## References
* https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
* https://gist.github.com/n1try/af0b8476ae4106ec098fea1dfe57f578
* http://www.derongliu.org/adp/adp-cdrom/Barto1983.pdf
