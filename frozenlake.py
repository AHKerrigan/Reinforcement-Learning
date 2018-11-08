import gym
import numpy as np
env = gym.make('FrozenLake8x8-v0')
env.render()

Q_Table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.85
gamma = 0.99
episodes = 5000
results = []
game_over = 0
all_rewards = 0


for episode in range(episodes):

	# Reset the enviroment 
	state = env.reset()
	game_over = False

	while game_over != True:
		# env.render()

		# Choose an action from the Q table
		# The last term adds noise that decays over furhter episodes
		# This way, theres more exploration near the beginning
		action = np.argmax(Q_Table[state] + np.random.randn(1, env.action_space.n) * (1./(episode+1)))

		new_state, reward, game_over, _ = env.step(action)

		Q_Table[state, action] = Q_Table[state, action] + alpha * (reward + gamma * np.max(Q_Table[new_state]) - Q_Table[state, action])


		all_rewards += reward
		state = new_state

	results.append(all_rewards)
print(results)