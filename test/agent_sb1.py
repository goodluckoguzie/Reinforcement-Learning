import time

from simplejson import load
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

import socnavenv
from socnavenv import SocNavEnv

env = SocNavEnv()
# action_noise = NormalActionNoise(mean=np.array([0.0, 0.6]), sigma=np.array([0.2, 0.1]))
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0.6, 0.0]), sigma=np.array([0.2, 0.1]), theta=1)
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1,  tensorboard_log='./logs/')

# net_arch = [128, 128, dict(vf=[128, 128], pi=[128, 128])]
episode_len = 10000
total_reward = 0
try:
    model = DDPG.load("ppo_socnavenv")
    loaded_from_file = True
except:
    loaded_from_file = False

if not loaded_from_file:
    # Learn and save
    model.learn(total_timesteps=100000000)
    model.save("ppo_socnavenv")
    for i in range(20):
        print('DOOOOOOOOONEEEEEEEE')



def evaluate_policy(env, policy, n_episodes=100):
    total_rewards = 0.0
    for _ in range(n_episodes):
        total_rewards += run_episode(env, policy)
    return total_rewards / n_episodes

def gen_random_policy():
    #return np.random.choice(4, size=((16)))
    action, _states = model.predict(obs, deterministic=True)


def crossover(policy1, policy2):
    new_policy = policy1.copy()
    for i in range(16):
        rand = np.random.uniform()
        if rand > 0.5:
            new_policy[i] = policy2[i]
    return new_policy

def mutation(policy, p=0.05):
    new_policy = policy.copy()
    for i in range(16):
        rand = np.random.uniform()
        if rand < p:
            new_policy[i] = np.random.choice(4)
    return new_policy

env.close()






if __name__ == '__main__':
    np.random.seed(1234)
 
    n_policy = 100
    n_steps = 20
    start = time.time()
    policy_pop = [gen_random_policy() for _ in range(n_policy)]
    for idx in range(n_steps):
        policy_scores = [evaluate_policy(env, p) for p in policy_pop]
        print('Generation %d : max score = %0.2f' %(idx+1, max(policy_scores)))
        policy_ranks = list(reversed(np.argsort(policy_scores)))
        elite_set = [policy_pop[x] for x in policy_ranks[:5]]
        select_probs = np.array(policy_scores) / np.sum(policy_scores)
        child_set = [crossover(
            policy_pop[np.random.choice(range(n_policy), p=select_probs)], 
            policy_pop[np.random.choice(range(n_policy), p=select_probs)])
            for _ in range(n_policy - 5)]
        mutated_list = [mutation(p) for p in child_set]
        policy_pop = elite_set
        policy_pop += mutated_list
    policy_score = [evaluate_policy(env, p) for p in policy_pop]
    best_policy = policy_pop[np.argmax(policy_score)]

    end = time.time()
    print('Best policy score = %0.2f. Time taken = %4.4f'
            %(np.max(policy_score), (end-start)))    
