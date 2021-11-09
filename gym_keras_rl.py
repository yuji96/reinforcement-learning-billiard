import gym

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy


env = gym.make('CartPole-v0')

window_length = 1
input_shape = (window_length,) + env.observation_space.shape
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

model.summary()

memory = SequentialMemory(limit=50000, window_length=window_length)
policy = BoltzmannQPolicy()
agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
agent.compile(Adam())


import matplotlib.pyplot as plt

# fit の結果を取得しておく
history = agent.fit(env, nb_steps=10000, visualize=False, verbose=1)
# agent.test(env, nb_episodes=5, visualize=True)

# 結果を表示
plt.subplot(2,1,1)
plt.plot(history.history["nb_episode_steps"])
plt.ylabel("step")

plt.subplot(2,1,2)
plt.plot(history.history["episode_reward"])
plt.xlabel("episode")
plt.ylabel("reward")

plt.show()  # windowが表示されます。