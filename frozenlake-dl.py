import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import gym

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def create_model():
	model = Sequential()
	model.add(Dense(10, input_dim=64 , activation='relu'))
	model.add(Dense(10, activation="relu"))
	model.add(Dense(4))
	model.compile(loss='mse', optimizer=Adam())
	return model


def OH(state_space, x):
	state_m = np.zeros((1, state_space))
	state_m[0][x] = 1
	return state_m


env = gym.make('FrozenLake8x8-v0')

episodes = 10000
gamma = 0.99
epsilon = 0.1
reward_count = 0
model = create_model()
steps = 100
total_reward = 0
reward_history = []

for episode in range(episodes):
	state = env.reset()
	
	for step in range(steps):

		# Get out Q values for the current state
		Q = model.predict(OH(64, state), batch_size=1)

		# 10% of the time, we'll just take a random action
		if np.random.rand(1) < epsilon:
			action = env.action_space.sample()
		else:
			action = np.argmax(Q)

		# Take the action
		new_state, reward, game_over, _ = env.step(action)

		# Use our model to get the predicted maximum future reward 
		# given our new action		
		Q_Future = model.predict(OH(64, new_state), batch_size=1)
		maxQ_future = np.max(Q_Future)

		# Bellman Equation
		targetQ = Q
		targetQ[0, action] = reward + (gamma * maxQ_future)

		# Train on target Q vector
		history = model.fit(OH(64, state), targetQ, verbose=False, batch_size=1)

		# Add reward tyo our memory
		total_reward += reward
		state = new_state

		if game_over == True:
			if reward == 1:
				epsilon = 1/((episode/50) + 10)
				print("We got a reward at episode ", episode, "!")
			break
	
	reward_history.append(total_reward)

plt.figure()
plt.plot(pd.Series(reward_history))
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()
		