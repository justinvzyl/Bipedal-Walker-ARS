import gym 
import pybullet_envs
import pybullet
import numpy as np
from multiprocessing import Pool
import time


"""
Class to contain all the hyperparameters used by the ARS algorithm
 
Description of the hyperparameters:
    step_size --> learning rate 
    N --> number of random samples to generate
    noise_std --> the standard deviation of the noise used on the random samples
    num_max_training_steps --> number of iterations for the ARS algorithm
    seed --> the random number seed 
    env_name --> the gym environment for the pybullet version of Walker2D
    agents --> number of cpus for multiprocessing
    chunk_size --> number of chunks to divide data into for agents
    
"""
class hyper_paramaters():
    
    def __init__(self,
                 step_size = 0.03,
                 N = 32,            
                 noise_std = 0.025,
                 num_max_training_steps = 1000,
                 seed = 132,
                 env_name = 'Walker2DBulletEnv-v0',
                 #env_name = 'HalfCheetahBulletEnv-v0',
                 agents = 8,
                 chunk_size = 4,
                 ):
        self.step_size = step_size
        self.N = N
        self.noise_std = noise_std
        self.n_training_steps = num_max_training_steps
        self.seed = seed
        self.env_name = env_name
        self.agents = agents
        self.chunk_size = chunk_size
"""
Class to normalize the data and keep a running mean as well as a running
which is later used for the standard deviation

Methods:
    __init__(self,n_inputs)
        creates holding variables;
        
        n = number of inputs (observations), 
        mean = running mean, 
        mean_diff = the spread of new values, 
        var = the variance of inputs
    
    observe(self,x)
        updates the running mean, mean_diff, and variance
        cliping the variance ensures no division by zero occurs
        
    normalize(self, inputs)
        normalizes the input data (observations) to ensure all states are
        weighed equally
        
        returns normalized inputs
"""

class Normalizer():
    def __init__(self, n_inputs):
        self.n = n_inputs
        self.mean = np.zeros(n_inputs)
        self.mean_diff = np.zeros(n_inputs)
        self.var = np.zeros(n_inputs)
        
        
    def observe(self, x):
        self.n += 1
        last_mean = self.mean.copy()
        self.mean += (x-self.mean) / self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
        
        
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs  - obs_mean) / obs_std
"""
Function to run an episode using a provided policy and returns the cumalative 
reward for the episode

Paramaters: alldata --> tuple containing the environment name, policy, episode 
                        length, the normalizer, 
                        
Main mechanics: gets observation from environment, calculates the action using 
                the policy, take a step in the environment using action and
                recieve reward and next state. Repeat till done
                
Returns: total_reward --> total cumalative reward
"""   
def evaluation(alldata):
        env_name = alldata[0]
        policy = alldata[1]
        episode_length = alldata[2]
        normalizer = alldata[3]
        env = gym.make(env_name)
        observation = env.reset()
        
        done = False
        total_reward = 0
        length = 0

        while not done and length < episode_length:
            normalizer.observe(observation)
            observation = normalizer.normalize(observation)
            action = np.dot(observation,policy)
            
            observation, reward, done, _ = env.step(action)
            
            reward = max(min(reward,1),-1)
            total_reward += reward
            length += 1
            
        return total_reward

"""
Function creates two datasets, one for policies plus the random noise and one 
for policies minus the random noise

Paramaters: env --> environment name
            policy --> current policy
            deltas --> random samples
            episode_length --> maximum number of steps per episode
            normalizer --> normalizer object for running tallies

Returns: dataset_p --> policy+deltas
         dataset_m --> policy-deltas
                        
"""
def create_dataset(env, policy, deltas, episode_length,normalizer):
    dataset_p = [(env, policy+delta, episode_length,normalizer) for delta in deltas]
    dataset_m = [(env, policy-delta, episode_length,normalizer) for delta in deltas]
    
    return dataset_p,dataset_m




"""
Main ARS alogrithm
Creates the hyperparameters, initializes the policy, creates the normalizer
object, sets the episode length, sets the number of best policies to keep and 
ensures that the keep amount is not more than the amount of policies generated.


"""
    
if __name__ == '__main__':
    hp = hyper_paramaters()
    env = gym.make(hp.env_name)
    policy = np.zeros(shape=(env.observation_space.shape[0],env.action_space.shape[0]))
    normalizer = Normalizer(env.observation_space.shape[0])
    episode_length = 2000
    num_best_deltas = 20 
    assert num_best_deltas <= hp.N
    np.random.seed(hp.seed)

    j = 0
    
    #Main loop for ARS training, number of training iterations is less than training_steps
    while j < hp.n_training_steps: 
        start = time.time()
        r_plus = []
        r_minus = []
        deltas = [np.random.randn(*policy.shape) for _ in range(hp.N)]
        deltas_noise = [delta*hp.noise_std for delta in deltas] 
        dataset_p,dataset_m = create_dataset(hp.env_name,policy,deltas_noise,episode_length,normalizer)
        
        #Evaluate policy/deltas using multi processing
        with Pool(processes = hp.agents) as pool:
           r_plus = pool.map(evaluation, dataset_p,hp.chunk_size)
           r_minus = pool.map(evaluation, dataset_m, hp.chunk_size)
           
        #Sorts rewards, and compiles final rollouts
        std_rewards = np.array(r_plus+r_minus).std()
        rewards = {k:max(rplus,rminus) for k,(rplus,rminus) in enumerate(zip(r_plus,r_minus))}
        sorted_rewards = sorted(rewards.keys(), key=lambda x:rewards[x], reverse = True)[:num_best_deltas]
        rollout_rewards = [(r_plus[k], r_minus[k], deltas[k]) for k in sorted_rewards]
       
        #calculates update rule
        diff_reward_delta = np.zeros(policy.shape)
        for sample_num in range(num_best_deltas):
            diff_reward_delta += (rollout_rewards[sample_num][0]-rollout_rewards[sample_num][1])*rollout_rewards[sample_num][2]
        
        old_policy = policy
        policy = old_policy + hp.step_size / (num_best_deltas*std_rewards) * diff_reward_delta
        j += 1
        end = time.time()
        #Saves policy every 50 steps as a new policy
        if j%50 == 0 and j is not 0:
            np.savetxt('policy_'+str(j)+'.csv',policy,delimiter=',')
        f1 = open('./log.txt','a+')
        score = evaluation([hp.env_name, policy, episode_length, normalizer])
        f1.write('Iteration: %d\t Current Reward: %.2f\t Std Rewards: %.2f\t Time per iteration: %.4f\n'% (j, score, std_rewards, end-start))
        f1.close()
 
            
    


        
        
        
        



