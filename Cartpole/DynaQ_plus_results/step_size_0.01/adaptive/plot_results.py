import pylab as plt
import numpy as np

ps_0 = np.load('./planning_step_0.npy')
ps_5 = np.load('./planning_step_5.npy')
ps_10 = np.load('./planning_step_10.npy')
ps_50 = np.load('./planning_step_50.npy')
SCORES = [sum(i)/len(i) for i in [ps_0, ps_5, ps_10, ps_50]]

plt.figure(figsize=(50, 6))
plt.title('Dyna-Q: various planning steps')
plt.plot(ps_0)
plt.plot(ps_5)
plt.plot(ps_10)
plt.plot(ps_50)
plt.legend(('planning step 0', 'planning step 5', 'planning step 10', 'planning step 50'), loc='best')
plt.axhline(y=np.max(SCORES), color='black', linestyle='--')
print("The max score is {} from index {}.".format(np.max(SCORES), np.argmax(SCORES)))

plt.show()