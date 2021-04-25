import gym

import billiard_gym  # noqa

env = gym.make('billiard-v0')
observation = env.reset()

max_iter = 100

for i in range(max_iter):
    action = [0.16656725990615653]
    # action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    if not done:
        print(reward, done)

    if done:
        env.reset()
