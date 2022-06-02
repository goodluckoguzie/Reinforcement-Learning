from statistics import mode
import numpy as np
import random
import time
import gym
from gym import wrappers

def run_episode(env, policy, episode_len=100):
    total_reward = 0
    obs = env.reset()

    for t in range(episode_len):
        #env.render()
        action = policy[obs]
        #action = np.array([0.95, 0.01])
        #action = OrnsteinUhlenbeckActionNoise(mean=np.array([0.6, 0.0]), sigma=np.array([0.2, 0.1]), theta=1)

        print('action now')
        print(action)
        print('action1 now')
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            # print('Epside finished after {} timesteps.'.format(t+1))
            break
    return total_reward


def evaluate_policy(env, policy, n_episodes=100):
    total_rewards = 0.0
    for _ in range(n_episodes):
        total_rewards += run_episode(env, policy)
    return total_rewards / n_episodes

def gen_random_policy():
    #return np.random.choice(4, size=((16)))

    #action, _states = model.predict(obs, deterministic=True)
    #action = policy[obs]
    action = np.array([0.95, 0.01])
    print ('action')
    print(action)
    print('action1')
    return #model.predict(obs, deterministic=True)


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

if __name__ == '__main__':
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
    print(action_noise)
    print('action_noise')
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1,  tensorboard_log='./logs/')



    # net_arch = [128, 128, dict(vf=[128, 128], pi=[128, 128])]
    episode_len = 1#10000
    total_reward = 0
    try:
        model = DDPG.load("ppo_socnavenv1")
        loaded_from_file = True
    except:
        loaded_from_file = False

    if not loaded_from_file:
        # Learn and save
        model.learn(total_timesteps=1)#00000000)
        model.save("ppo_socnavenv1")
        for i in range(2):
            print('DOOOOOOOOONEEEEEEEE')


    ## Policy search
    n_policy = 1#100
    n_steps =1
    start = time.time()
    policy_pop = [gen_random_policy() for _ in range(n_policy)]
    #print( policy_pop)

    for idx in range(n_steps):
        policy_scores = [evaluate_policy(env, p) for p in policy_pop]
        print( policy_scores)


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

    ## Evaluation
    #env = wrappers.Monitor(env, '/tmp/frozenlake1', force=True)
    #for _ in range(200):
        #run_episode(env, best_policy)
    #Senv.close()
    #gym.upload('/tmp/frozenlake1', api_key=...)

