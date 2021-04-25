from billiards import Billiard
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from config import Ball, CUSHIONS, INITIAL_WHITE_POS, LENGTH, RADIUS, WIDTH


class Simulator(Billiard):
    def __init__(self, cue_ball=Ball.WHITE):
        super().__init__(obstacles=CUSHIONS)
        self.position_list = []
        self.cue_ball = cue_ball
        self.is_success = False
        self.previous_ball = None
        self.is_collided_2ball = False
        self.cushion_count = 0
        self.last_cushion_time = None

    def collide_balls(self, idx1, idx2):
        if self.cue_ball in {idx1, idx2}:
            another_ball = idx1 if idx1 != self.cue_ball else idx2
            if not self.previous_ball:
                assert idx1 != idx2
                self.previous_ball = another_ball
            elif self.previous_ball != another_ball:
                self.is_collided_2ball = True
        return super().collide_balls(idx1, idx2)

    def collide_obstacle(self, idx, obs):
        if idx == self.cue_ball:
            self.cushion_count += 1
            if self.cushion_count == 3:
                self.last_cushion_time = self.time
        return super().collide_obstacle(idx, obs)

    def _move(self, time):
        super()._move(time)
        pos = {"time": time}
        pos.update({i: coords.copy() for i, coords in enumerate(self.balls_position)})
        self.position_list.append(pos)

    def simulate(self, end_time=10, fps=30):
        frames = int(fps * end_time)
        for i in range(frames):
            self.evolve(i / fps)

        position_df = pd.DataFrame(self.position_list)
        self.is_success = self.is_collided_2ball and (self.cushion_count >= 3)
        return position_df


class BilliardEnv(gym.Env):
    """強化学習用のビリヤード環境.

    Attributes
    ----------
    metadata : dict
        メタデータ.
    action_space : gym.spaces.Space
        行動(Action)の張る空間.行動の種類.
    observation_space : gym.spaces.Space
        観測値(Observation)の張る空間.報酬の種類.
    simulator : utils.Simulator
        ビリヤード力学系エンジン.

    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=0, high=2 * np.pi, shape=(1,), dtype=np.float32)

        low = np.array([0] * 6 + [-200] * 2)
        high = np.array([LENGTH] * 6 + [200] * 2)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.simulator = None
        self.position_df = None
        self.reset()

    def reset(self):
        """状態を初期化し、初期の観測値を返す."""
        self.relocate()
        return self.observation

    def step(self, action):
        """actionを実行し、結果を返す."""
        rad, *_ = action
        self.relocate(rad)

        self.position_df = self.simulator.simulate()
        return self.observation, self.reward, self.done, {}

    def render(self, mode="human"):
        fig, ax = plt.subplots(1, 1)
        for i in range(3):
            ax.plot(*zip(*self.position_df[i]))
            ax.scatter(*self.position_df[i].iat[0], marker="o")
        ax.set_xlim([0, LENGTH])
        ax.set_ylim([0, WIDTH])
        plt.show()

    @property
    def observation(self):
        pos = np.array(self.simulator.balls_position).flatten()
        vel = np.array(self.simulator.balls_velocity).flatten()[4:]
        return np.hstack([pos, vel])

    @property
    def reward(self, success_reward=100, intercept=50, scale=15):
        out = 0
        d = self.calc_min_distance()

        out += int(self.simulator.is_success) * success_reward
        out += intercept * np.exp(-d / scale)

        return out

    @property
    def done(self):
        return not self.simulator.is_success

    def relocate(self, rad=None):
        if rad is None:
            yellow_pos = (LENGTH * 0.25, WIDTH * 0.5)
            red_pos = (LENGTH * 0.75, WIDTH * 0.5)
        else:
            yellow_pos, red_pos = self.simulator.balls_position[:2]

        self.simulator = Simulator()
        for pos in [yellow_pos, red_pos]:
            self.simulator.add_ball(pos, (0, 0), RADIUS)
        self.set_white_ball(rad)

    def set_white_ball(self, rad):
        if self.simulator.balls_position.shape[0] == 3:
            white_pos = self.simulator.balls_position[2]
        else:
            white_pos = INITIAL_WHITE_POS

        rad = rad if (rad is not None) else random.uniform(0, 360)

        direction = rotate(rad)
        speed = 60
        vec = direction / np.linalg.norm(direction) * speed
        self.simulator.add_ball(white_pos, vec, RADIUS)

    def calc_min_distance(self):
        position_df = self.position_df
        sim = self.simulator
        if not sim.last_cushion_time:
            return np.inf
        last_ball = [i for i in range(3) if i not in {sim.cue_ball, sim.previous_ball}][0]
        df = position_df.loc[position_df["time"] > sim.last_cushion_time, [sim.cue_ball, last_ball]]
        distances = df.apply(lambda x: np.linalg.norm(x.iat[0] - x.iat[1]) - 2 * RADIUS, axis=1)
        return distances.min()


def rotate(rad, vec=np.array([1, 0])):
    R = np.array([[np.cos(rad), -np.sin(rad)],
                  [np.sin(rad),  np.cos(rad)]])
    return R @ vec
