import time
import numpy as np
from test import SocNavEnv

env = SocNavEnv()


while True:
    obs = env.reset()
    print('episode')
    while not env.done:
        action = np.array([0.5, 0.1])
        obs, reward, done, info = env.step(0)
        time.sleep(0.05)

