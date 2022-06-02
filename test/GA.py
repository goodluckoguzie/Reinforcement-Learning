import gym
# Selecting environments
#env01 = gym.make('MountainCar-v0')
#env02 = gym.make('MountainCarContinuous-v0')

import time

from simplejson import load
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

import socnavenv
from socnavenv import SocNavEnv

env01 = SocNavEnv()

# Print Observation Space details -----------------------
print("** Observation Space Details **")
print("MountainCar-v0, Observation Space--\n" +str(env01.observation_space))
print("Observation Space high:" +str(env01.observation_space.high))
print("Observation Space low:" +str(env01.observation_space.low))

# Print Action Space space detail ------------------------
print("** Action Space Details **")
print("MountainCar-v0, Action Space--\n"+str(env01.action_space))
