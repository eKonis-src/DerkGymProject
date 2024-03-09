from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
import math
import os.path

env = DerkEnv(
  home_team=[
    {'primaryColor': '#ff00ff', 'slots': ['Pistol', 'ParalyzingDart', 'FrogLegs'],
     'rewardFunction': {'damageEnemyStatue': 10, 'damageEnemyUnit': 6, 'timeSpentHomeTerritory': 10,
                        'timeSpentAwayTerritory': -10, 'fallDamageTaken': -40}},
    {'primaryColor': '#00ff00', 'slots': ['BloodClaws', 'Shell', 'IronBubblegum'],
     'rewardFunction': {'damageEnemyStatue': 10, 'damageEnemyUnit': 6, 'timeSpentAwayTerritory': 10,
                        'timeSpentHomeTerritory': -10, 'fallDamageTaken': -40}},
    {'primaryColor': '#ff0000', 'slots': [None, 'HeliumBubblegum', 'VampireGland'],
     'rewardFunction': {'healTeammate1': 3, 'healEnemy': -10, 'fallDamageTaken': -40}}
  ],
  away_team=[
    {'primaryColor': '#ff00ff', 'slots': ['Pistol', 'ParalyzingDart', 'FrogLegs'],
     'rewardFunction': {'damageEnemyStatue': 10, 'damageEnemyUnit': 6, 'timeSpentHomeTerritory': 10,
                        'timeSpentAwayTerritory': -10, 'fallDamageTaken': -40}},
    {'primaryColor': '#00ff00', 'slots': ['BloodClaws', 'Shell', 'IronBubblegum'],
     'rewardFunction': {'damageEnemyStatue': 10, 'damageEnemyUnit': 6, 'timeSpentAwayTerritory': 10,
                        'timeSpentHomeTerritory': -10, 'fallDamageTaken': -40}},
    {'primaryColor': '#ff0000', 'slots': [None, 'HeliumBubblegum', 'VampireGland'],
     'rewardFunction': {'healTeammate1': 3, 'healEnemy': -10, 'fallDamageTaken': -40}}
  ],
  turbo_mode=False
)


class Network:
  def __init__(self, weights=None, biases=None):
    self.network_outputs = 13
    if weights is None:
      weights_shape = (self.network_outputs, len(ObservationKeys))
      self.weights = np.random.normal(size=weights_shape)
    else:
      self.weights = weights
    if biases is None:
      self.biases = np.random.normal(size=(self.network_outputs))
    else:
      self.biases = biases

  def clone(self):
    return Network(np.copy(self.weights), np.copy(self.biases))

  def forward(self, observations):
    outputs = np.add(np.matmul(self.weights, observations), self.biases)
    casts = outputs[3:6]
    cast_i = np.argmax(casts)
    focuses = outputs[6:13]
    focus_i = np.argmax(focuses)
    return (
      math.tanh(outputs[0]),  # MoveX
      math.tanh(outputs[1]),  # Rotate
      max(min(outputs[2], 1), 0),  # ChaseFocus
      (cast_i + 1) if casts[cast_i] > 0 else 0,  # CastSlot
      (focus_i + 1) if focuses[focus_i] > 0 else 0,  # Focus
    )

  def copy_and_mutate(self, network, mr=0.1):
    self.weights = np.add(network.weights, np.random.normal(size=self.weights.shape) * mr)
    self.biases = np.add(network.biases, np.random.normal(size=self.biases.shape) * mr)


weights = np.load('weights.npy') if os.path.isfile('weights.npy') else None
biases = np.load('biases.npy') if os.path.isfile('biases.npy') else None

networks = [Network(weights, biases) for i in range(env.n_agents)]
#env.mode = 'train'
total_reward = 0


for e in range(10):
  observation_n = env.reset()
  while True:
    action_n = [networks[i].forward(observation_n[i]) for i in range(env.n_agents)]
    observation_n, reward_n, done_n, info = env.step(action_n)
    if all(done_n):
        print("Episode finished")
        break
  if env.mode == 'train':
    reward_n = env.total_reward

    print(reward_n)
    top_network_i = np.argmax(reward_n)
    top_network = networks[top_network_i].clone()
    for network in networks:
      network.copy_and_mutate(top_network)
    print('top reward', reward_n[top_network_i])
    np.save('weights.npy', top_network.weights)
    np.save('biases.npy', top_network.biases)
    total_reward += reward_n[top_network_i]
print(total_reward / 30)

env.close()
