import time
import numpy as np
from socnavenv import SocNavEnv

env = SocNavEnv()


while True:
    obs = env.reset()
    print('episode')
    while not env.done:
        action = np.array([0.95, 0.01])
        obs, reward, done, info = env.step(action)
        time.sleep(0.001)

