from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color
from skimage.transform import downscale_local_mean

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
image_shape = (60, 64, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=image_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Flatten())
model.add(Dense(7, activation='linear'))
model.compile(loss='mse', optimizer=Adam(clipnorm=10, lr=1e-4, decay=1e-6, epsilon=1e-4))


"""
done = True
for step in range(5000):
	if done:
		state = env.reset()
	state, reward, done, info = env.step(env.action_space.sample())
	print(env.action_space.samp)

env.close()
"""
episodes = 1000
gamma = 0.99
epsilon = 0.1
reward_count = 0
reward_history = []
max_episode_reward = 0
episode_reward = 0
batch_size = 32

def process_image(state):
	state = color.rgb2gray(state)
	state = np.resize(downscale_local_mean(state, (4, 4)), (1, 60, 64, 1))
	return state

def process_batch(batch):
	inputs = []
	targets = []
	for item in batch:
		state = item[0]
		action = item[1]
		reward = item[2]
		Q = item[3]
		new_state = item[4]

		# Use our model to get the predicted maximum future reward 
		# given our new action	
		Q_Future = model.predict(new_state, batch_size=1)
		maxQ_future = np.max(Q_Future)

		# Bellman Equation
		targetQ = Q
		targetQ[0, action] = reward + (gamma * maxQ_future)

		inputs.append(np.reshape(state, (60, 64, 1)))
		targets.append(targetQ)

	inputs = np.asarray(inputs)
	targets = np.reshape(np.asarray(targets), (batch_size, 7))
	model.fit(inputs, targets, batch_size=batch_size, verbose=False)


# The memory batch we'll build, as well as the counter that will
# keep track of how big our memory is 
batch = []
batch_count = 0
for episode in range(episodes):
	state = env.reset()
	state = process_image(state)
	episode_reward = 0
	while(True):
		env.render()

		# Get out Q values for the current state
		Q = model.predict(state, batch_size=1)
		print(Q)

		# 10% of the time, we'll just take a random action
		if np.random.rand(1) < epsilon:
			#print("Taking random action")
			action = env.action_space.sample()
		else:
			#print("Taking non-random action")
			action = np.argmax(Q)

		# Take the action
		new_state, reward, game_over, _ = env.step(action)
		new_state = process_image(new_state)
		batch.append([state, action, reward, Q, new_state])
		batch_count += 1
		if batch_count == batch_size:
			process_batch(batch)
			batch = []
			batch_count = 0

		# Explore more as we get further closer and closer toward our 
		# Current best score 
		# This way we stick with what works, and only explore when we get closer to
		# "uncharted territory"
		#epsilon = 0.4 * ((reward + 300)/(max_episode_reward + 300))
		episode_reward += reward
		state = new_state

		# Go to the next episode if we die
		if game_over==True:
			break

	if episode_reward > max_episode_reward:
		print("New best score!", episode_reward, "at episode", episode, "!")
		max_episode_reward = episode_reward
	else:
		print("Score at episode", episode, ":", episode_reward)
	reward_history.append(episode_reward)

plt.figure()
plt.plot(pd.Series(reward_history))
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()
