import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import billiard_gym  # noqa
from utils import Buffer, OUActionNoise, get_actor, get_critic, policy, update_target
print("end import")


def train(env_info, buffer, total_episodes=100, noise_std=0.2, gamma=0.99, tau=0.005):
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(noise_std) * np.ones(1))

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    for ep in range(total_episodes):

        prev_state = env.reset()
        episodic_reward = 0

        while True:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(actor_model, tf_prev_state, ou_noise, **env_info)
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)

            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward

            buffer.learn(gamma)
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            print(f"-- Episode: {ep} "+"-"*20)
            print(f"Reward: {reward}\tDone: {done}\nState: {state}")

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        # print("Episode * {ep} * Avg Reward is ==> {avg_reward}")
        avg_reward_list.append(avg_reward)

    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.savefig("reward.png")

    # Save the weights
    # actor_model.save_weights(".model/pendulum_actor.h5")
    # critic_model.save_weights(".model/pendulum_critic.h5")
    # target_actor.save_weights(".model/pendulum_target_actor.h5")
    # target_critic.save_weights(".model/pendulum_target_critic.h5")


if __name__ == "__main__":
    env = gym.make("billiard-v0")
    env_info = {
        "num_states": env.observation_space.shape,
        "num_actions": env.action_space.shape,
        "upper_bound": env.action_space.high[0],
        "lower_bound": env.action_space.low[0],
    }

    actor_model = get_actor(**env_info)
    critic_model = get_critic(**env_info)
    target_actor = get_actor(**env_info)
    target_critic = get_critic(**env_info)

    # Learning rate for actor-critic models
    critic_lr = 0.002
    actor_lr = 0.001
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    buffer = Buffer(actor_model, critic_model, target_actor, target_critic,
                    critic_optimizer, actor_optimizer,
                    buffer_capacity=50000, batch_size=64, **env_info)

    train(env_info, buffer, total_episodes=10000)
