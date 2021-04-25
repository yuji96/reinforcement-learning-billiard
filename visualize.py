import billiards
import gym
import matplotlib.pyplot as plt
import numpy as np

import billiard_gym  # noqa
from billiard_gym.envs import Simulator, rotate


def generate_image(thetas):
    for i, theta in enumerate(thetas):
        observation = env.reset()
        observation, reward, done, _ = env.step([theta])
        env.render(f"results/{i}")


def generate_movie(thetas):
    width, length = 112, 224
    radius = 2.85
    for i, theta in enumerate(thetas):
        bld = Simulator()
        vel = rotate(theta, vec=np.array([200, 0]))

        bld.add_ball((length * 0.25, width * 0.5), (0, 0), radius)
        bld.add_ball((length * 0.75, width * 0.5), (0, 0), radius)
        bld.add_ball((length * 0.25, width * 3/8), vel, radius)

        anim = billiards.visualize.animate(bld, end_time=3)
        anim._fig.set_size_inches((10, 5.5))
        anim.save(f"results/{i}.mp4")
        # plt.show()

if __name__ == "__main__":
    thetas = [0.04074701, 0.07725584, 0.09100453, 0.09105028, 0.09153747, 0.09163393, 0.04081714, 0.07705076]
    env = gym.make('billiard-v0')
    generate_movie(thetas)
