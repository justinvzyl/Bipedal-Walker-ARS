# Bipedal-Walker-ARS

Implementation of the Augmented Random Search (ARS) ML algorithm to the Walker2DBulletEnv-v0 environment. This specific implementation uses multiprocessing to divide up the workload of policy evaluations. 

This implementation reaches stability of the maximum reward after around 200 ARS iterations with 64 episodes per rollout. 

Dependencies:

OpenAI Gym
PyBullet
Numpy
Multiprocessing

Usage:

Python3 ars_algo.py 


