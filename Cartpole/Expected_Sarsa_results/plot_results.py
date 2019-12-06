import pylab as plt
import numpy as np

s0 = np.load('./step_size_0.01.npy')
s1 = np.load('./step_size_0.05.npy')
s2 = np.load('./step_size_0.1.npy')
s3 = np.load('./step_size_0.5.npy')
SCORES = [sum(i)/len(i) for i in [s0, s1, s2, s3]]

plt.figure(figsize=(50, 6))
plt.title('Expected Sarsa: various step size')
plt.plot(s0)
plt.plot(s1)
plt.plot(s2)
plt.plot(s3)
plt.legend(('step_size_0.01', 'step_size_0.05', 'step_size_0.1', 'step_size_0.5'), loc='best')
plt.axhline(y=np.max(SCORES), color='black', linestyle='--')
print("The max score is {} from index {}.".format(np.max(SCORES), np.argmax(SCORES)))

plt.show()