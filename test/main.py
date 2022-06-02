import time
import gym
import numpy as np
import stable_baselines3
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC
import socnavenv
from socnavenv import SocNavEnv
from simplejson import load

env = SocNavEnv()
#model = SAC(MlpPolicy, env, verbose=1,tensorboard_log='./logs/').learn(10000)
model = SAC.load("application/zip/sac_socnavenv")
"""
try:
    model = SAC.load("sac_socnavenv")
    loaded_from_file = True
    print('True')
except:
    loaded_from_file = False
    print('False')

if not loaded_from_file:
    # Learn and save
    model.learn(total_timesteps=50000, log_interval=10)
    model.save("sac_socnavenv")
    for i in range(5):
        print('DOOOOOOOOONEEEEEEEE')

    #del model # remove to demonstrate saving and loading
    #model = SAC.load("sac_pendulum")
"""

obs = env.reset()
#while True:
for i in range(100000000000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

    time.sleep(socnavenv.TIMESTEP)
    if done:
      obs = env.reset()

env.close()


#####################################################################################################################
